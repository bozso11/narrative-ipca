"""Mathematical tests for ``narrative_ipca.sparse_ipca``.

Panels are built directly with :class:`narrative_ipca.types.IPCAPanel` from a
known factor structure (``r = c Gamma f + noise``), so every check is against
brute force, a closed form, an invariance or the known truth.

The solver works in standardised instrument coordinates (``Gamma~ = diag(sigma_c)
Gamma``, an exact reparametrisation of Eq. 8; module docstring of ``sparse_ipca``).
Tests that replay the loop (``_first_gamma_step``,
``test_obj_path_entries_are_the_eq8_objective``) do so in those coordinates,
because the start point (SVD of the standardised managed portfolios) and hence
the iterates and ``lambda_max`` depend on them; everything they assert is stated
in original units. The last section checks the reparametrisation itself: unit
invariance, the objective identity and the conditioning of the Gamma-step.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from narrative_ipca.config import EstimationConfig, LambdaGridConfig
from narrative_ipca.grouplasso import group_lasso_gram
from narrative_ipca.types import IPCAPanel, compute_sigma_c
from narrative_ipca import grouplasso as gl
from narrative_ipca import sparse_ipca as si

TRUE_ROWS = (2, 5, 9)  # rows of Gamma (1-based narrative index +1: topics 1, 4, 8)

BACKENDS = [
    pytest.param("numpy", id="numpy"),
    pytest.param("numba", id="numba", marks=pytest.mark.skipif(not gl.HAVE_NUMBA, reason="numba not available (not installed or disabled by NARRATIVE_IPCA_NO_NUMBA)")),
]


@pytest.fixture(params=BACKENDS)
def backend(request, monkeypatch):
    """Run the fit on one Gamma-step sweep implementation (numba kernel or numpy reference)."""
    monkeypatch.setattr(gl, "USE_NUMBA", request.param == "numba")
    assert gl.active_backend() == request.param
    return request.param


# ---------------------------------------------------------------------------
# synthetic panel with known structure
# ---------------------------------------------------------------------------
def make_panel(
    N: int = 60,
    T: int = 40,
    L: int = 12,
    K: int = 2,
    true_rows: tuple[int, ...] = TRUE_ROWS,
    noise: float = 0.3,
    seed: int = 0,
    empty_period: int | None = None,
    drop_frac: float = 0.0,
) -> tuple[IPCAPanel, np.ndarray, np.ndarray]:
    """``r_{i,t} = c_{i,t-1} Gamma f_t + noise`` with ``len(true_rows)`` relevant instruments."""
    rng = np.random.default_rng(seed)
    Gamma = np.zeros((L + 1, K))
    Gamma[0] = np.array([0.5, 0.2, -0.1])[:K]
    for l in true_rows:
        Gamma[l] = rng.standard_normal(K)
    mu = np.array([0.3, 0.1, 0.05])[:K]
    sd = np.array([1.0, 0.7, 0.5])[:K]
    F = mu + sd * rng.standard_normal((T, K))
    X = np.ones((N * T, L + 1))
    X[:, 1:] = rng.standard_normal((N * T, L))
    t_idx = np.repeat(np.arange(T), N)
    a_idx = np.tile(np.arange(N), T)
    y = np.einsum("nk,nk->n", X @ Gamma, F[t_idx]) + noise * rng.standard_normal(N * T)
    keep = np.ones(N * T, dtype=bool)
    if empty_period is not None:
        keep &= t_idx != empty_period
    if drop_frac > 0.0:
        keep &= rng.random(N * T) > drop_frac
    periods = pd.date_range("2000-01-31", periods=T, freq="D")
    panel = IPCAPanel(
        X=X[keep],
        y=y[keep],
        t_idx=t_idx[keep],
        asset_idx=a_idx[keep],
        periods=periods,
        assets=np.array([f"a{i}" for i in range(N)]),
        instrument_names=["const"] + [f"topic_{l}" for l in range(L)],
        sigma_c=compute_sigma_c(X[keep]),
    )
    return panel, Gamma, F


def subspace_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between col(A) and col(B)."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)


def true_mask(L: int = 12) -> np.ndarray:
    m = np.zeros(L, dtype=bool)
    m[[r - 1 for r in TRUE_ROWS]] = True
    return m


@pytest.fixture(scope="module")
def base():
    panel, Gamma, F = make_panel()
    cfg = EstimationConfig(K=2, tol=1e-10, max_iter=1000)
    lam_max = si.lambda_max(panel, cfg)
    return panel, Gamma, F, cfg, lam_max


# ---------------------------------------------------------------------------
# building blocks against brute force
# ---------------------------------------------------------------------------
def test_f_step_matches_ridge_regression_per_period():
    panel, _, _ = make_panel(N=25, T=8, empty_period=3)
    mom = panel.moments()
    rng = np.random.default_rng(7)
    Gamma = rng.standard_normal((panel.p, 2))
    F = si.f_step(mom.S, mom.V, Gamma)  # ridge = 2 (Eq. 16)
    F0 = si.f_step(mom.S, mom.V, Gamma, ridge=0.0)
    for t, sl in panel.period_slices():
        if sl.stop <= sl.start:
            np.testing.assert_array_equal(F[t], 0.0)
            np.testing.assert_array_equal(F0[t], 0.0)
            continue
        B = panel.X[sl] @ Gamma
        y = panel.y[sl]
        ref = np.linalg.solve(B.T @ B + 2.0 * np.eye(2), B.T @ y)
        np.testing.assert_allclose(F[t], ref, rtol=1e-10, atol=1e-12)
        # ridge = 2 minimises 0.5||y - B f||^2 + ||f||^2 (check against perturbations)
        obj = lambda f: 0.5 * np.sum((y - B @ f) ** 2) + f @ f
        for _ in range(20):
            assert obj(F[t] + 1e-3 * rng.standard_normal(2)) >= obj(F[t]) - 1e-12
        ref0 = np.linalg.lstsq(B, y, rcond=None)[0]
        np.testing.assert_allclose(F0[t], ref0, rtol=1e-8, atol=1e-10)
    assert F.shape == (panel.T, 2)


def test_gram_from_moments_matches_kron_design():
    panel, _, _ = make_panel(N=15, T=6, L=5, true_rows=(1, 3), empty_period=2)
    mom = panel.moments()
    rng = np.random.default_rng(3)
    F = rng.standard_normal((panel.T, 3))
    G, b = si.gram_from_moments(mom.S, mom.V, F)
    D = np.stack([np.kron(panel.X[i], F[panel.t_idx[i]]) for i in range(panel.n_obs)])
    np.testing.assert_allclose(G, D.T @ D, rtol=1e-11, atol=1e-9)
    np.testing.assert_allclose(b, D.T @ panel.y, rtol=1e-11, atol=1e-9)
    assert G.shape == (panel.p * 3, panel.p * 3)
    # coordinate map: x[l*K + k] = Gamma[l, k] reproduces c' Gamma f
    Gamma = rng.standard_normal((panel.p, 3))
    fitted = D @ Gamma.reshape(-1)
    np.testing.assert_allclose(fitted, si.fitted_values(panel, Gamma, F), rtol=1e-12)


def test_objective_value_matches_long_form_formula():
    panel, _, _ = make_panel(N=20, T=10, empty_period=4)
    rng = np.random.default_rng(5)
    Gamma = rng.standard_normal((panel.p, 2))
    F = rng.standard_normal((panel.T, 2))
    lam = 0.37
    resid = panel.y - si.fitted_values(panel, Gamma, F)
    mom = panel.moments()
    assert si.ssr_from_moments(mom.S, mom.V, mom.yy, Gamma, F) == pytest.approx(resid @ resid, rel=1e-10)
    ref = 0.5 * resid @ resid + lam * panel.n_obs * panel.sigma_c @ np.linalg.norm(Gamma, axis=1) + np.sum(F**2)
    assert si.objective_value(panel, Gamma, F, lam, panel.sigma_c) == pytest.approx(ref, rel=1e-10)
    ref_no_int = ref - lam * panel.n_obs * 1.0 * np.linalg.norm(Gamma[0])
    assert si.objective_value(panel, Gamma, F, lam, panel.sigma_c, penalize_intercept=False) == pytest.approx(ref_no_int, rel=1e-10)
    pen = si.penalty_vector(lam, panel.n_obs, panel.sigma_c, penalize_intercept=False)
    assert pen[0] == 0.0 and np.all(pen[1:] == lam * panel.n_obs * panel.sigma_c[1:])


def test_betas_and_fitted_values_explicit_loops():
    panel, Gamma, F = make_panel(N=5, T=4, L=3, true_rows=(1, 2))
    B = si.betas(panel.X, Gamma)
    np.testing.assert_allclose(B, panel.X @ Gamma)
    fitted = si.fitted_values(panel, Gamma, F)
    for i in range(panel.n_obs):
        assert fitted[i] == pytest.approx(panel.X[i] @ Gamma @ F[panel.t_idx[i]], rel=1e-12)
    with pytest.raises(ValueError):
        si.betas(panel.X, Gamma[:-1])
    with pytest.raises(ValueError):
        si.fitted_values(panel, Gamma, F[:-1])


def test_objective_invariant_to_orthogonal_rotation_only():
    """D24: Eq. 8 is invariant to Gamma -> Gamma Q, f -> Q' f for orthogonal Q, not for general R."""
    panel, Gamma, F = make_panel(N=20, T=10)
    lam = 0.2
    base = si.objective_value(panel, Gamma, F, lam, panel.sigma_c)
    Q, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((2, 2)))
    assert si.objective_value(panel, Gamma @ Q, F @ Q, lam, panel.sigma_c) == pytest.approx(base, rel=1e-12)
    R = np.array([[2.0, 0.3], [0.0, 0.5]])
    assert si.objective_value(panel, Gamma @ R, F @ np.linalg.inv(R).T, lam, panel.sigma_c) != pytest.approx(base, rel=1e-6)


# ---------------------------------------------------------------------------
# the ARLS fit
# ---------------------------------------------------------------------------
def test_objective_non_increasing_along_arls(base, backend):
    panel, _, _, cfg, lam_max = base
    for frac in (0.02, 0.1):
        res = si.fit_sparse_ipca(panel, cfg, lam=frac * lam_max)
        path = np.asarray(res.obj_path)
        assert len(path) == res.n_iter + 1
        assert np.all(np.diff(path) <= 1e-9 * np.abs(path[:-1]))
        assert res.objective == pytest.approx(path[-1])
        assert res.objective == pytest.approx(si.objective_value(panel, res.Gamma, res.F, res.lam, panel.sigma_c), rel=1e-12)
        assert res.converged


def test_penalty_equals_twice_ridge_at_stationary_point(base, backend):
    """Scale stationarity of Eq. 8: lam N_S sum sigma_l ||Gamma_l|| = 2 sum_t ||f_t||^2."""
    panel, _, _, cfg, lam_max = base
    for frac, pen_int in ((0.02, True), (0.1, True), (0.05, False)):
        res = si.fit_sparse_ipca(panel, cfg, lam=frac * lam_max, penalize_intercept=pen_int)
        pen = si.penalty_vector(res.lam, panel.n_obs, panel.sigma_c, pen_int)
        P = pen @ res.gamma_norms
        R = np.sum(res.F**2)
        assert R > 0
        assert P / (2.0 * R) == pytest.approx(1.0, abs=1e-3)


def test_selection_recovers_true_rows_and_subspace(base):
    panel, Gamma_true, _, cfg, lam_max = base
    res = si.fit_sparse_ipca(panel, cfg, lam=0.02 * lam_max)
    assert res.converged
    truth = true_mask()
    assert np.array_equal(res.selected, truth), f"selected {np.where(res.selected)[0]} vs true {np.where(truth)[0]}"
    assert res.n_selected == 3
    cos = subspace_cos(res.Gamma_tilde[truth], Gamma_true[1:][truth])
    assert np.all(cos > 0.99), cos
    assert res.total_r2 > 0.85 and res.pred_r2 > 0.0
    assert np.array_equal(res.selected, res.gamma_norms[1:] > 0)
    assert res.mve_sharpe() > 0.0
    # the fit reproduces the true factor space up to rotation: regress true F on fitted F
    Fh = res.F
    coef = np.linalg.lstsq(np.c_[np.ones(len(Fh)), Fh], base[2], rcond=None)[0]
    Fpred = np.c_[np.ones(len(Fh)), Fh] @ coef
    r2 = 1 - np.sum((base[2] - Fpred) ** 2) / np.sum((base[2] - base[2].mean(0)) ** 2)
    assert r2 > 0.95


def test_lambda_max_zeroes_narrative_rows(base):
    panel, _, _, cfg, lam_max = base
    assert np.isfinite(lam_max) and lam_max > 0
    for mult in (1.0, 1.05):
        res = si.fit_sparse_ipca(panel, cfg, lam=mult * lam_max)
        assert res.n_selected == 0
        assert np.all(res.Gamma[1:] == 0.0)
    assert si.fit_sparse_ipca(panel, cfg, lam=0.02 * lam_max).n_selected > 0


def _first_gamma_step(panel, cfg, lam, pen_int):
    """Narrative block of the first penalised Gamma-step at ``lam`` given the warm-up factors.

    Replayed in the standardised coordinates the fit uses (moments ``S~``, ``V~``, uniform
    penalties); zero rows of ``Gamma~`` are zero rows of ``Gamma``, so the selection read off
    here is coordinate-free.
    """
    mom_t = si.standardized_moments(panel.moments(), panel.sigma_c)
    _, F, _ = si._warm_start(mom_t, cfg.K, cfg)
    G, b = si.gram_from_moments(mom_t.S, mom_t.V, F)
    pen = si.standardized_penalties(lam, panel.n_obs, panel.p, penalize_intercept=pen_int)
    x, info = group_lasso_gram(G, b, cfg.K, pen, max_iter=20_000, tol=1e-13)
    assert info["converged"]
    Gm = si.from_standardized(x.reshape(panel.p, cfg.K), panel.sigma_c)
    return Gm[0], Gm[1:]


@pytest.mark.parametrize("pen_int", [False, True])
def test_lambda_max_is_the_kkt_threshold_given_warmup_factors(base, pen_int):
    """D22: with the warm-up factors fixed, the first Gamma-step at lam_max has an all-zero narrative
    block and at a slightly smaller lambda it does not, for both intercept-penalty settings (the
    default penalises the intercept, D19, so the intercept-only solution is the shrunk one)."""
    panel, _, _, cfg, _ = base
    cfg_p = replace(cfg, penalize_intercept=pen_int)
    lam_max = si.lambda_max(panel, cfg_p)
    assert np.isfinite(lam_max) and lam_max > 0
    for mult, expect_zero in ((1.0 + 1e-9, True), (0.98, False)):
        _, narr = _first_gamma_step(panel, cfg_p, mult * lam_max, pen_int)
        assert np.all(narr == 0.0) == expect_zero, (pen_int, mult)


def test_lambda_max_differs_between_intercept_penalty_settings(base):
    """Ignoring the intercept penalty is not exact: the two settings give different thresholds and the
    unpenalised-intercept value applied under the default leaves the KKT test failing on one side."""
    panel, _, _, cfg, lam_pen = base  # base cfg penalises the intercept
    lam_unpen = si.lambda_max(panel, replace(cfg, penalize_intercept=False))
    assert lam_unpen != pytest.approx(lam_pen, rel=1e-3)
    # under the default the intercept row is already dead at lam_max here (regime 1 of the docstring)
    x0, narr = _first_gamma_step(panel, cfg, (1.0 + 1e-9) * lam_pen, True)
    assert np.all(x0 == 0.0) and np.all(narr == 0.0)
    fails = 0
    for mult in ((1.0 + 1e-9), 0.98):
        _, narr = _first_gamma_step(panel, cfg, mult * lam_unpen, True)
        fails += int(np.all(narr == 0.0) != (mult > 1.0))
    assert fails >= 1


def test_lambda_max_with_intercept_alive_at_threshold():
    """Regime 2 of the docstring: a strong common component keeps the penalised intercept alive at the
    threshold, and lam_max is still exactly the KKT switch point of the first Gamma-step."""
    panel, _, F_true = make_panel(N=40, T=30, seed=11)
    y = panel.y + 3.0 * F_true[panel.t_idx, 0] + 1.5  # large period-common return component
    strong = IPCAPanel(X=panel.X, y=y, t_idx=panel.t_idx, asset_idx=panel.asset_idx, periods=panel.periods,
                       assets=panel.assets, instrument_names=panel.instrument_names, sigma_c=panel.sigma_c)
    cfg = EstimationConfig(K=2, tol=1e-10, max_iter=500)
    lam_max = si.lambda_max(strong, cfg)
    x0, narr = _first_gamma_step(strong, cfg, (1.0 + 1e-9) * lam_max, True)
    assert np.any(x0 != 0.0), "intercept should survive at the threshold in this panel"
    assert np.all(narr == 0.0)
    _, narr = _first_gamma_step(strong, cfg, 0.98 * lam_max, True)
    assert np.any(narr != 0.0)


def test_intercept_only_solution_matches_group_lasso_solver():
    rng = np.random.default_rng(9)
    for K in (1, 2, 3):
        D = rng.standard_normal((2 * K + 1, K)) * np.array([1.0, 4.0, 0.3])[:K]  # unequal scales, PSD block
        G00 = D.T @ D
        b0 = G00 @ rng.standard_normal(K)  # in range(G00)
        for pen in (0.0, 0.1 * np.linalg.norm(b0), 0.9 * np.linalg.norm(b0)):
            x = si._intercept_only_solution(G00, b0, pen)
            ref, info = group_lasso_gram(G00, b0, K, np.array([pen]), max_iter=200_000, tol=1e-12)
            assert info["converged"]
            np.testing.assert_allclose(x, ref, atol=1e-7, rtol=1e-6)
        assert np.all(si._intercept_only_solution(G00, b0, 1.01 * np.linalg.norm(b0)) == 0.0)


def test_lambda_grid_and_values_override(base):
    panel, _, _, cfg, lam_max = base
    grid = si.lambda_grid(panel, replace(cfg, lam_grid=LambdaGridConfig(n_lambdas=7, ratio=1e-2)))
    assert grid.shape == (7,)
    assert grid[-1] == pytest.approx(lam_max, rel=1e-12)
    assert grid[0] == pytest.approx(1e-2 * lam_max, rel=1e-12)
    assert np.all(np.diff(grid) > 0)
    np.testing.assert_allclose(np.diff(np.log(grid)), np.diff(np.log(grid))[0])
    fixed = si.lambda_grid(panel, replace(cfg, lam_grid=LambdaGridConfig(values=(0.3, 0.1, 0.2))))
    np.testing.assert_array_equal(fixed, [0.1, 0.2, 0.3])


def test_lambda_path_ascending_warm_started(base):
    panel, _, _, cfg, lam_max = base
    lams = lam_max * np.array([0.3, 0.02, 0.1, 1.0])
    calls = []
    points, fits = si.lambda_path(panel, cfg, lams=lams, progress=lambda i, n, pt: calls.append((i, n, pt.lam)))
    assert [p.lam for p in points] == sorted(lams.tolist())
    assert len(fits) == len(points) == 4
    assert calls == [(i, 4, p.lam) for i, p in enumerate(points)]
    assert points[0].n_selected == 3 and points[-1].n_selected == 0
    assert points[0].n_selected >= points[1].n_selected >= points[2].n_selected >= points[3].n_selected
    for p, f in zip(points, fits):
        assert p.K == 2 and f.lam == p.lam
        assert p.total_r2 == f.total_r2 and p.objective == f.objective and p.n_iter == f.n_iter
        assert p.mve_sharpe == pytest.approx(f.mve_sharpe())
        np.testing.assert_array_equal(p.gamma_norms, f.gamma_norms)
    assert fits[0].meta["init"] == "svd+warmup" and fits[1].meta["init"] == "warm"
    # a warm-started path fit and a cold fit at the same lambda reach the same optimum
    cold = si.fit_sparse_ipca(panel, cfg, lam=points[1].lam)
    assert cold.objective == pytest.approx(fits[1].objective, rel=1e-6)
    assert np.array_equal(cold.selected, fits[1].selected)
    assert np.all(np.diff(points[0].total_r2 - np.array([p.total_r2 for p in points])) >= -1e-12)


def test_warm_start_reproduces_and_validation(base):
    panel, _, _, cfg, lam_max = base
    lam = 0.05 * lam_max
    res = si.fit_sparse_ipca(panel, cfg, lam=lam)
    again = si.fit_sparse_ipca(panel, cfg, lam=lam, Gamma_init=res.Gamma)
    assert again.meta["init"] == "warm" and "n_warmup" not in again.meta
    assert again.n_iter <= 3
    assert again.objective == pytest.approx(res.objective, rel=1e-8)
    np.testing.assert_allclose(again.Gamma, res.Gamma, atol=1e-5, rtol=1e-4)
    np.testing.assert_array_equal(again.selected, res.selected)
    # determinism given the seed
    rep = si.fit_sparse_ipca(panel, cfg, lam=lam)
    np.testing.assert_array_equal(rep.Gamma, res.Gamma)
    np.testing.assert_array_equal(rep.F, res.F)
    with pytest.raises(ValueError):
        si.fit_sparse_ipca(panel, cfg)  # cfg.lam is None and no lam given
    with pytest.raises(ValueError):
        si.fit_sparse_ipca(panel, cfg, lam=-1.0)
    with pytest.raises(ValueError):
        si.fit_sparse_ipca(panel, cfg, lam=lam, Gamma_init=np.zeros((panel.p, 3)))
    # cfg.lam is used when lam is not passed
    via_cfg = si.fit_sparse_ipca(panel, replace(cfg, lam=lam))
    assert via_cfg.lam == lam and via_cfg.objective == pytest.approx(res.objective)


def test_penalize_intercept_false_keeps_intercept_row(base):
    panel, _, _, cfg, lam_max = base
    res = si.fit_sparse_ipca(panel, cfg, lam=1.5 * lam_max, penalize_intercept=False)
    assert res.n_selected == 0
    assert res.gamma_norms[0] > 0.0
    assert res.meta["penalties"][0] == 0.0
    # with the intercept penalised the same lambda zeroes everything (Eq. 8 sum from l = 0)
    res2 = si.fit_sparse_ipca(panel, cfg, lam=1.5 * lam_max, penalize_intercept=True)
    assert np.all(res2.Gamma == 0.0) and np.all(res2.F == 0.0)
    assert res2.total_r2 == 0.0 and res2.mve_sharpe() == 0.0


def test_unbalanced_panel_with_empty_period():
    panel, Gamma_true, _ = make_panel(seed=3, empty_period=7, drop_frac=0.2)
    cfg = EstimationConfig(K=2, tol=1e-9, max_iter=1000)
    lam_max = si.lambda_max(panel, cfg)
    res = si.fit_sparse_ipca(panel, cfg, lam=0.03 * lam_max)
    assert res.converged
    assert not res.populated[7] and np.all(res.F[7] == 0.0)
    assert res.populated.sum() == panel.T - 1
    Fp = res.F[res.populated]
    np.testing.assert_allclose(res.mu_f, Fp.mean(axis=0))
    np.testing.assert_allclose(res.Sigma_ff, np.cov(Fp, rowvar=False, ddof=1))
    assert 0.0 < res.total_r2 < 1.0
    assert np.array_equal(res.selected, true_mask())
    # R2 definitions from the long form
    resid = panel.y - si.fitted_values(panel, res.Gamma, res.F)
    assert res.total_r2 == pytest.approx(1 - resid @ resid / (panel.y @ panel.y), rel=1e-8)
    resid_p = panel.y - si.betas(panel.X, res.Gamma) @ res.mu_f
    assert res.pred_r2 == pytest.approx(1 - resid_p @ resid_p / (panel.y @ panel.y), rel=1e-8)


# ---------------------------------------------------------------------------
# canonical form
# ---------------------------------------------------------------------------
def test_canonicalize_is_an_invariant_relabelling(base):
    panel, _, _, cfg, lam_max = base
    res = si.fit_sparse_ipca(panel, cfg, lam=0.05 * lam_max)
    can = si.canonicalize(res)
    assert can.objective == res.objective  # copied
    assert si.objective_value(panel, can.Gamma, can.F, res.lam, panel.sigma_c) == pytest.approx(res.objective, rel=1e-10)
    np.testing.assert_allclose(si.fitted_values(panel, can.Gamma, can.F), si.fitted_values(panel, res.Gamma, res.F), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(can.gamma_norms, res.gamma_norms, rtol=1e-10)
    np.testing.assert_array_equal(can.selected, res.selected)
    np.testing.assert_array_equal(can.Gamma[1:][~res.selected], 0.0)
    off = can.Sigma_ff - np.diag(np.diag(can.Sigma_ff))
    assert np.max(np.abs(off)) < 1e-12 * np.max(np.diag(can.Sigma_ff))
    assert np.all(np.diff(np.diag(can.Sigma_ff)) <= 0)
    assert np.all(can.mu_f >= 0)
    assert can.mve_sharpe() == pytest.approx(res.mve_sharpe(), rel=1e-10)
    np.testing.assert_allclose(np.sum(can.F**2), np.sum(res.F**2), rtol=1e-12)
    # consistency of the rotated moments with the rotated factors
    Fp = can.F[can.populated]
    np.testing.assert_allclose(can.mu_f, Fp.mean(0), atol=1e-12)
    np.testing.assert_allclose(can.Sigma_ff, np.cov(Fp, rowvar=False, ddof=1), atol=1e-12)
    # idempotent
    twice = si.canonicalize(can)
    np.testing.assert_allclose(twice.Gamma, can.Gamma, atol=1e-12)
    np.testing.assert_allclose(twice.F, can.F, atol=1e-12)


# ---------------------------------------------------------------------------
# plain IPCA (lambda = 0)
# ---------------------------------------------------------------------------
def test_fit_ipca_theta_y_normalisation_and_column_space(base):
    panel, Gamma_true, F_true, cfg, _ = base
    res = si.fit_ipca(panel, 2, cfg)
    assert res.converged and res.lam == 0.0
    assert np.all(res.selected) and res.selected.shape == (panel.L,)
    np.testing.assert_allclose(res.Gamma.T @ res.Gamma, np.eye(2), atol=1e-12)
    Fp = res.F[res.populated]
    M = Fp.T @ Fp / len(Fp)
    assert abs(M[0, 1]) < 1e-12 * M[0, 0]
    assert M[0, 0] >= M[1, 1] > 0
    for k in range(2):
        col = res.Gamma[:, k]
        assert col[np.argmax(np.abs(col) > 1e-12 * np.abs(col).max())] > 0
    # column space of the full Gamma and of its relevant rows
    assert np.all(subspace_cos(res.Gamma, Gamma_true) > 0.99)
    assert np.all(subspace_cos(res.Gamma_tilde[true_mask()], Gamma_true[1:][true_mask()]) > 0.99)
    # SSR path non-increasing; objective is the SSR of the reported pair
    path = np.asarray(res.obj_path)
    assert np.all(np.diff(path) <= 1e-9 * path[:-1])
    resid = panel.y - si.fitted_values(panel, res.Gamma, res.F)
    assert res.objective == pytest.approx(resid @ resid, rel=1e-8)
    assert res.meta["objective_kind"] == "ssr"
    # F is the exact (unridged) F-step of the reported Gamma
    np.testing.assert_allclose(res.F, si.f_step(panel.moments().S, panel.moments().V, res.Gamma, ridge=0.0), atol=1e-8)
    # fit_sparse_ipca(lam=0) dispatches here (D18)
    via = si.fit_sparse_ipca(panel, cfg, lam=0.0, K=2)
    np.testing.assert_array_equal(via.Gamma, res.Gamma)
    assert via.lam == 0.0


def test_fit_ipca_handles_collinear_instruments():
    """D10a: a redundant instrument (simplex-like dependence) makes G singular; pinv keeps the fit."""
    panel, Gamma_true, _ = make_panel(N=40, T=30, L=6, true_rows=(1, 3, 5))
    X = panel.X.copy()
    X[:, -1] = -X[:, 1:-1].sum(axis=1)  # rows of the narrative block now sum to zero
    dep = panel.with_instruments(X, panel.instrument_names)
    cfg = EstimationConfig(K=2, tol=1e-10, max_iter=500)
    res = si.fit_ipca(dep, 2, cfg)
    assert np.all(np.isfinite(res.Gamma)) and res.converged
    np.testing.assert_allclose(res.Gamma.T @ res.Gamma, np.eye(2), atol=1e-10)
    assert np.all(np.diff(np.asarray(res.obj_path)) <= 1e-9 * np.asarray(res.obj_path[:-1]))
    assert res.total_r2 > 0.5


# ---------------------------------------------------------------------------
# the f-step projections and the objective read from them
# ---------------------------------------------------------------------------
def test_f_step_parts_match_definitions_and_ssr_from_moments():
    """``A_t = Gamma' S_t Gamma``, ``rhs_t = Gamma' V_t`` and the SSR from them equals the moment SSR."""
    panel, _, _ = make_panel(N=25, T=9, empty_period=2, drop_frac=0.1)
    mom = panel.moments()
    rng = np.random.default_rng(17)
    Gamma = rng.standard_normal((panel.p, 2))
    F, A, rhs = si._f_step_parts(mom.S, mom.V, Gamma)
    np.testing.assert_allclose(F, si.f_step(mom.S, mom.V, Gamma), rtol=1e-14, atol=1e-16)
    for t in range(panel.T):
        np.testing.assert_allclose(A[t], Gamma.T @ mom.S[t] @ Gamma, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(rhs[t], Gamma.T @ mom.V[t], rtol=1e-12, atol=1e-12)
    sum_yy = float(np.sum(mom.yy))
    # at the f-step F and at an arbitrary F
    for Fm in (F, rng.standard_normal((panel.T, 2))):
        ref = si.ssr_from_moments(mom.S, mom.V, mom.yy, Gamma, Fm)
        assert si._ssr_from_parts(sum_yy, A, rhs, Fm) == pytest.approx(ref, rel=1e-12)
        resid = panel.y - si.fitted_values(panel, Gamma, Fm)
        assert si._ssr_from_parts(sum_yy, A, rhs, Fm) == pytest.approx(resid @ resid, rel=1e-10)
    # empty input
    F0, A0, r0 = si._f_step_parts(np.zeros((0, panel.p, panel.p)), np.zeros((0, panel.p)), Gamma)
    assert F0.shape == (0, 2) and A0.shape == (0, 2, 2) and r0.shape == (0, 2)


def test_obj_path_entries_are_the_eq8_objective(base, backend):
    """Every obj_path entry (read from the f-step projections) equals objective_value at that sweep.

    The loop is replayed in the standardised coordinates the fit uses (start-point dependent, so
    the replay must follow the same coordinates); every entry is then compared with the
    original-unit ``objective_value`` of the unscaled ``Gamma`` (the O(T p^2) reference).
    """
    panel, _, _, cfg, lam_max = base
    lam = 0.05 * lam_max
    res = si.fit_sparse_ipca(panel, cfg, lam=lam)
    mom_t = si.standardized_moments(panel.moments(), panel.sigma_c)
    Gamma_t, _, _ = si._warm_start(mom_t, cfg.K, cfg)
    F = si.f_step(mom_t.S, mom_t.V, Gamma_t)
    pen = si.standardized_penalties(lam, panel.n_obs, panel.p, True)
    unscale = lambda Gt: si.from_standardized(Gt, panel.sigma_c)
    assert res.obj_path[0] == pytest.approx(si.objective_value(panel, unscale(Gamma_t), F, lam, panel.sigma_c), rel=1e-12)
    for k in range(1, min(len(res.obj_path), 6)):
        G, b = si.gram_from_moments(mom_t.S, mom_t.V, F)
        x, _ = group_lasso_gram(G, b, cfg.K, pen, x0=Gamma_t.ravel(), max_iter=cfg.inner_max_iter, tol=cfg.inner_tol)
        Gamma_t = x.reshape(panel.p, cfg.K)
        F = si.f_step(mom_t.S, mom_t.V, Gamma_t)
        assert res.obj_path[k] == pytest.approx(si.objective_value(panel, unscale(Gamma_t), F, lam, panel.sigma_c), rel=1e-12)


# ---------------------------------------------------------------------------
# both Gamma-step backends give the same fit
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not gl.HAVE_NUMBA, reason="numba not available (not installed or disabled by NARRATIVE_IPCA_NO_NUMBA)")
@pytest.mark.parametrize("frac,pen_int", [(0.02, True), (0.1, True), (0.05, False)])
def test_backends_agree_on_sparse_ipca_fit(base, monkeypatch, frac, pen_int):
    panel, _, _, cfg, lam_max = base
    out = {}
    for name in ("numpy", "numba"):
        monkeypatch.setattr(gl, "USE_NUMBA", name == "numba")
        out[name] = si.fit_sparse_ipca(panel, cfg, lam=frac * lam_max, penalize_intercept=pen_int)
    a, b = out["numba"], out["numpy"]
    assert a.n_iter == b.n_iter and a.converged == b.converged
    assert a.meta["inner_iters"] == b.meta["inner_iters"]
    assert np.array_equal(a.selected, b.selected)
    assert a.objective == pytest.approx(b.objective, rel=1e-12)
    np.testing.assert_allclose(a.obj_path, b.obj_path, rtol=1e-12)
    scale = np.max(np.abs(b.Gamma))
    np.testing.assert_allclose(a.Gamma, b.Gamma, rtol=0.0, atol=1e-11 * scale)
    np.testing.assert_allclose(a.F, b.F, rtol=0.0, atol=1e-11 * np.max(np.abs(b.F)))
    assert a.mve_sharpe() == pytest.approx(b.mve_sharpe(), rel=1e-10)


@pytest.mark.skipif(not gl.HAVE_NUMBA, reason="numba not available (not installed or disabled by NARRATIVE_IPCA_NO_NUMBA)")
def test_backends_agree_on_lambda_max_and_path(base, monkeypatch):
    panel, _, _, cfg, _ = base
    cfg_p = replace(cfg, lam_grid=LambdaGridConfig(n_lambdas=5, ratio=0.02))
    out = {}
    for name in ("numpy", "numba"):
        monkeypatch.setattr(gl, "USE_NUMBA", name == "numba")
        lmax = si.lambda_max(panel, cfg_p)
        points, fits = si.lambda_path(panel, cfg_p)
        out[name] = (lmax, points, fits)
    assert out["numba"][0] == pytest.approx(out["numpy"][0], rel=1e-12)
    for pa, pb in zip(out["numba"][1], out["numpy"][1]):
        assert pa.lam == pytest.approx(pb.lam, rel=1e-12)
        assert pa.n_selected == pb.n_selected and pa.n_iter == pb.n_iter
        assert pa.objective == pytest.approx(pb.objective, rel=1e-12)
        assert pa.mve_sharpe == pytest.approx(pb.mve_sharpe, rel=1e-10)
        np.testing.assert_allclose(pa.gamma_norms, pb.gamma_norms, rtol=1e-10, atol=1e-14)


# ---------------------------------------------------------------------------
# standardised instrument coordinates (numerical reparametrisation, 2026-09-06)
# ---------------------------------------------------------------------------
def make_ill_panel(
    N: int = 200,
    T: int = 120,
    L: int = 40,
    K: int = 2,
    true_rows: tuple[int, ...] = (2, 7, 15, 23),
    noise: float = 0.3,
    seed: int = 0,
    log10_scales: tuple[float, float] = (-6.0, -4.0),
    rho: float = 0.8,
) -> tuple[IPCAPanel, np.ndarray, np.ndarray]:
    """A panel whose instrument columns look like kernel covariances: scales log-spaced between
    ``10**log10_scales``, non-zero means, and a ``K``-factor structure shared across topics (the
    columns are ``beta_i' a_l`` plus noise, the D47 mechanism), next to the constant column of 1.
    Returns are ``r = c Gamma f + noise`` with ``Gamma`` supported on ``true_rows`` in the units of
    the *unscaled* columns, so the true ``Gamma`` rows in the panel's units are ``Gamma_l / scale_l``."""
    rng = np.random.default_rng(seed)
    scales = np.logspace(log10_scales[0], log10_scales[1], L)
    beta = rng.standard_normal((N, K))
    a = rng.standard_normal((L, K))
    means = rng.uniform(0.5, 2.0, L)
    t_idx = np.repeat(np.arange(T), N)
    a_idx = np.tile(np.arange(N), T)
    Z = means[None, :] + rho * (beta[a_idx] @ a.T) + np.sqrt(1.0 - rho**2) * rng.standard_normal((N * T, L))
    Gamma = np.zeros((L + 1, K))
    Gamma[0] = np.array([0.5, 0.2, -0.1])[:K]
    for l in true_rows:
        Gamma[l] = rng.standard_normal(K)
    mu = np.array([0.3, 0.1, 0.05])[:K]
    sd = np.array([1.0, 0.7, 0.5])[:K]
    F = mu + sd * rng.standard_normal((T, K))
    X0 = np.column_stack([np.ones(N * T), Z])
    y = np.einsum("nk,nk->n", X0 @ Gamma, F[t_idx]) + noise * rng.standard_normal(N * T)
    X = X0.copy()
    X[:, 1:] *= scales[None, :]
    periods = pd.date_range("2000-01-31", periods=T, freq="D")
    panel = IPCAPanel(
        X=X,
        y=y,
        t_idx=t_idx,
        asset_idx=a_idx,
        periods=periods,
        assets=np.array([f"a{i}" for i in range(N)]),
        instrument_names=["const"] + [f"topic_{l}" for l in range(L)],
        sigma_c=compute_sigma_c(X),
    )
    return panel, Gamma / np.r_[1.0, scales][:, None], F


def rescaled_panel(panel: IPCAPanel, col: int, factor: float) -> IPCAPanel:
    """The same panel with instrument column ``col`` (>= 1) in other units (``sigma_c`` recomputed)."""
    X = panel.X.copy()
    X[:, col] *= factor
    return panel.with_instruments(X, panel.instrument_names)


def assert_close(a, b, rtol=1e-9):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = max(float(np.max(np.abs(b))), 1e-300)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * scale)


def test_standardized_coordinates_are_an_exact_reparametrisation():
    """The identities of the module docstring: ``S~ = D S D``, ``V~ = D V``, ``X Gamma = X~ Gamma~``,
    the f-step and its projections are invariant, ``G~ = (D kron I) G (D kron I)``, ``b~ = (D kron I) b``,
    the uniform penalty on ``Gamma~`` equals the ``sigma_l``-weighted one on ``Gamma``, and the group
    lasso objective of the Gamma-step is the same number in both coordinates."""
    panel, _, _ = make_panel(N=25, T=9, empty_period=2, drop_frac=0.1)
    mom = panel.moments()
    sigma = panel.sigma_c
    assert sigma[0] == 1.0 and np.all(sigma > 0)
    mom_t = si.standardized_moments(mom, sigma)
    D = np.diag(1.0 / sigma)
    for t in range(panel.T):
        np.testing.assert_allclose(mom_t.S[t], D @ mom.S[t] @ D, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(mom_t.V[t], D @ mom.V[t], rtol=1e-13, atol=1e-15)
    np.testing.assert_array_equal(mom_t.yy, mom.yy)
    np.testing.assert_array_equal(mom_t.n, mom.n)
    # the standardised instruments themselves have unit panel std
    Xt = panel.X @ D
    np.testing.assert_allclose(Xt[:, 1:].std(axis=0), 1.0, rtol=1e-12)
    rng = np.random.default_rng(11)
    Gamma = rng.standard_normal((panel.p, 2))
    Gamma_t = si.to_standardized(Gamma, sigma)
    np.testing.assert_allclose(si.from_standardized(Gamma_t, sigma), Gamma, rtol=1e-15)
    np.testing.assert_allclose(Xt @ Gamma_t, panel.X @ Gamma, rtol=1e-12)
    F, A, rhs = si._f_step_parts(mom.S, mom.V, Gamma)
    F_t, A_t, rhs_t = si._f_step_parts(mom_t.S, mom_t.V, Gamma_t)
    np.testing.assert_allclose(F_t, F, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(A_t, A, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(rhs_t, rhs, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(si.f_step(mom_t.S, mom_t.V, Gamma_t, ridge=0.0), si.f_step(mom.S, mom.V, Gamma, ridge=0.0), rtol=1e-9, atol=1e-11)
    G, b = si.gram_from_moments(mom.S, mom.V, F)
    G_t, b_t = si.gram_from_moments(mom_t.S, mom_t.V, F)
    DI = np.kron(D, np.eye(2))
    np.testing.assert_allclose(G_t, DI @ G @ DI, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(b_t, DI @ b, rtol=1e-12, atol=1e-12)
    lam = 0.3
    for pen_int in (True, False):
        pen = si.penalty_vector(lam, panel.n_obs, sigma, pen_int)
        pen_t = si.standardized_penalties(lam, panel.n_obs, panel.p, pen_int)
        assert np.all(pen_t[1:] == lam * panel.n_obs) and pen_t[0] == (lam * panel.n_obs if pen_int else 0.0)
        assert pen @ np.linalg.norm(Gamma, axis=1) == pytest.approx(pen_t @ np.linalg.norm(Gamma_t, axis=1), rel=1e-13)
        assert gl.group_lasso_objective(G, b, Gamma.ravel(), 2, pen) == pytest.approx(gl.group_lasso_objective(G_t, b_t, Gamma_t.ravel(), 2, pen_t), rel=1e-12)
        assert si.objective_value(panel, Gamma, F, lam, sigma, pen_int) == pytest.approx(
            0.5 * si.ssr_from_moments(mom_t.S, mom_t.V, mom_t.yy, Gamma_t, F) + pen_t @ np.linalg.norm(Gamma_t, axis=1) + np.sum(F**2), rel=1e-12
        )
    with pytest.raises(ValueError):
        si.standardized_moments(mom, np.r_[sigma[:-1], 0.0])
    with pytest.raises(ValueError):
        si.to_standardized(Gamma, sigma[:-1])
    bad = IPCAPanel(X=panel.X, y=panel.y, t_idx=panel.t_idx, asset_idx=panel.asset_idx, periods=panel.periods,
                    assets=panel.assets, instrument_names=panel.instrument_names, sigma_c=np.r_[sigma[:-1], 0.0])
    with pytest.raises(ValueError):
        si.fit_sparse_ipca(bad, EstimationConfig(K=2), lam=0.1)


@pytest.mark.parametrize("pen_int", [True, False])
def test_fit_is_invariant_to_the_units_of_an_instrument(base, backend, pen_int):
    """Rescaling instrument column ``l`` by 1e6 (``sigma_c`` recomputed) changes nothing but the units of
    row ``l`` of ``Gamma``: fitted values, objective, obj_path, R2s, selection, factors, row norms up to
    the rescaling, and ``Gamma`` after ``Gamma_l -> Gamma_l * 1e6`` agree to 1e-9 relative. (The previous
    original-unit solver moved by 5e-6 in fitted values on this panel and by 20% in ``F``/``Gamma`` on
    the full-size study panel, because its SVD start depends on the column scales.)"""
    panel_a, _, _, cfg, lam_max = base
    col, factor = 3, 1e6
    panel_b = rescaled_panel(panel_a, col, factor)
    assert panel_b.sigma_c[col] == pytest.approx(factor * panel_a.sigma_c[col], rel=1e-12)
    lam = 0.05 * lam_max
    ra = si.fit_sparse_ipca(panel_a, cfg, lam=lam, penalize_intercept=pen_int)
    rb = si.fit_sparse_ipca(panel_b, cfg, lam=lam, penalize_intercept=pen_int)
    assert ra.converged and rb.converged and ra.n_selected > 0
    assert_close(si.fitted_values(panel_b, rb.Gamma, rb.F), si.fitted_values(panel_a, ra.Gamma, ra.F))
    assert rb.objective == pytest.approx(ra.objective, rel=1e-9)
    assert len(rb.obj_path) == len(ra.obj_path)
    assert_close(rb.obj_path, ra.obj_path)
    assert rb.total_r2 == pytest.approx(ra.total_r2, rel=1e-9)
    assert rb.pred_r2 == pytest.approx(ra.pred_r2, rel=1e-9)
    assert rb.mve_sharpe() == pytest.approx(ra.mve_sharpe(), rel=1e-9)
    np.testing.assert_array_equal(rb.selected, ra.selected)
    assert_close(rb.F, ra.F)
    Gb = rb.Gamma.copy()
    Gb[col] *= factor
    assert_close(Gb, ra.Gamma)
    nb = rb.gamma_norms.copy()
    nb[col] *= factor
    assert_close(nb, ra.gamma_norms)
    # reported penalties are in original units (lam N_S sigma_l), the solver's are uniform
    np.testing.assert_allclose(ra.meta["penalties"], si.penalty_vector(lam, panel_a.n_obs, panel_a.sigma_c, pen_int), rtol=1e-12)
    assert rb.meta["penalties"][col] == pytest.approx(factor * ra.meta["penalties"][col], rel=1e-12)
    np.testing.assert_array_equal(ra.meta["penalties_tilde"], si.standardized_penalties(lam, panel_a.n_obs, panel_a.p, pen_int))
    np.testing.assert_array_equal(rb.meta["penalties_tilde"], ra.meta["penalties_tilde"])
    assert ra.meta["coordinates"] == "standardized"
    # warm starts convert on entry: a fit started from the other panel's (rescaled) solution reproduces it
    warm = ra.Gamma.copy()
    warm[col] /= factor
    rw = si.fit_sparse_ipca(panel_b, cfg, lam=lam, Gamma_init=warm, penalize_intercept=pen_int)
    assert rw.n_iter <= 3 and rw.objective == pytest.approx(ra.objective, rel=1e-8)
    np.testing.assert_array_equal(rw.selected, ra.selected)


@pytest.mark.parametrize("pen_int", [True, False])
def test_lambda_max_grid_and_path_are_unit_invariant(base, pen_int):
    panel_a, _, _, cfg, _ = base
    cfg_p = replace(cfg, penalize_intercept=pen_int, lam_grid=LambdaGridConfig(n_lambdas=4, ratio=0.02))
    col, factor = 5, 1e6
    panel_b = rescaled_panel(panel_a, col, factor)
    la, lb = si.lambda_max(panel_a, cfg_p), si.lambda_max(panel_b, cfg_p)
    assert lb == pytest.approx(la, rel=1e-9)
    np.testing.assert_allclose(si.lambda_grid(panel_b, cfg_p), si.lambda_grid(panel_a, cfg_p), rtol=1e-9)
    # (0.3 lam_max is left out: with the intercept unpenalised that point does not converge in
    # cfg.max_iter sweeps on either panel and costs ~10 s on the numpy backend; the invariance holds
    # there too, but a non-converged fit adds nothing to the check.)
    cfg_p = replace(cfg_p, max_iter=200)
    lams = la * np.array([0.02, 0.1, 1.0])
    pa, fa = si.lambda_path(panel_a, cfg_p, lams=lams)
    pb, fb = si.lambda_path(panel_b, cfg_p, lams=lams)
    assert pa[0].n_selected > 0 and pa[-1].n_selected == 0
    for qa, qb, ra, rb in zip(pa, pb, fa, fb):
        assert qb.n_selected == qa.n_selected and qb.n_iter == qa.n_iter and qb.converged == qa.converged
        assert qb.objective == pytest.approx(qa.objective, rel=1e-9)
        assert qb.total_r2 == pytest.approx(qa.total_r2, rel=1e-9)
        assert qb.mve_sharpe == pytest.approx(qa.mve_sharpe, rel=1e-9)
        nb = qb.gamma_norms.copy()
        nb[col] *= factor
        assert_close(nb, qa.gamma_norms)
        np.testing.assert_array_equal(rb.selected, ra.selected)
        assert_close(si.fitted_values(panel_b, rb.Gamma, rb.F), si.fitted_values(panel_a, ra.Gamma, ra.F))


def test_fit_ipca_is_unit_invariant_up_to_its_normalisation(base):
    """Plain IPCA in standardised coordinates: fitted values and the SSR agree to 1e-9; ``Gamma`` is
    reported with ``Gamma'Gamma = I_K`` in the *original* units of each panel, so the two ``Gamma`` agree
    as column spaces after undoing the rescaling (not entrywise: the normalisation is unit-dependent)."""
    panel_a, _, _, cfg, _ = base
    col, factor = 3, 1e6
    panel_b = rescaled_panel(panel_a, col, factor)
    ra, rb = si.fit_ipca(panel_a, 2, cfg), si.fit_ipca(panel_b, 2, cfg)
    assert ra.converged and rb.converged
    assert_close(si.fitted_values(panel_b, rb.Gamma, rb.F), si.fitted_values(panel_a, ra.Gamma, ra.F))
    assert rb.objective == pytest.approx(ra.objective, rel=1e-9)
    assert rb.total_r2 == pytest.approx(ra.total_r2, rel=1e-9)
    for r in (ra, rb):
        np.testing.assert_allclose(r.Gamma.T @ r.Gamma, np.eye(2), atol=1e-12)
    Gb = rb.Gamma.copy()
    Gb[col] *= factor
    assert np.all(subspace_cos(Gb, ra.Gamma) > 1.0 - 1e-9)
    assert not np.allclose(Gb, ra.Gamma, rtol=1e-6)  # the Theta_Y representative is not unit-invariant
    assert ra.meta["coordinates"] == "standardized"


@pytest.mark.parametrize("pen_int", [True, False])
def test_lambda_max_is_invariant_to_the_reparametrisation_given_the_factors(base, pen_int):
    """D22 threshold ``max_l ||b_l - G_{l0} x*_0|| / (N_S sigma_l)`` evaluated with the original-unit
    Gram equals the same threshold in standardised coordinates (``b~_l = b_l / sigma_l``, weight
    ``N_S``), and ``lambda_max`` equals both at the intercept-only solution it uses."""
    panel, _, _, cfg, _ = base
    cfg_p = replace(cfg, penalize_intercept=pen_int)
    lam_max = si.lambda_max(panel, cfg_p)
    K, L, n = cfg.K, panel.L, panel.n_obs
    mom = panel.moments()
    mom_t = si.standardized_moments(mom, panel.sigma_c)
    _, F, _ = si._warm_start(mom_t, K, cfg_p)  # the warm-up factors lambda_max conditions on

    def threshold(G, b, w, x0):
        resid = (b[K:] - G[K:, :K] @ x0).reshape(L, K)
        return float(np.max(np.linalg.norm(resid, axis=1) / w[1:]))

    G, b = si.gram_from_moments(mom.S, mom.V, F)
    G_t, b_t = si.gram_from_moments(mom_t.S, mom_t.V, F)
    w, w_t = n * panel.sigma_c, n * np.ones(panel.p)
    np.testing.assert_allclose(G_t[:K, :K], G[:K, :K], rtol=1e-12)  # sigma_0 = 1: same intercept block
    np.testing.assert_allclose(b_t[:K], b[:K], rtol=1e-12)
    pen0 = lam_max * n if pen_int else 0.0
    x0 = si._intercept_only_solution(G[:K, :K], b[:K], pen0)
    h, h_t = threshold(G, b, w, x0), threshold(G_t, b_t, w_t, x0)
    assert h_t == pytest.approx(h, rel=1e-10)
    assert lam_max == pytest.approx(h, rel=1e-8)
    # and at an arbitrary intercept the two formulas still agree
    rng = np.random.default_rng(2)
    xr = rng.standard_normal(K)
    assert threshold(G_t, b_t, w_t, xr) == pytest.approx(threshold(G, b, w, xr), rel=1e-10)


@pytest.mark.parametrize("pen_int", [True, False])
def test_reported_objective_equals_eq8_on_the_returned_pair(base, pen_int):
    """``objective`` / ``obj_path[-1]`` (evaluated in standardised coordinates during the loop) equal the
    original-unit Eq. 8 value of the returned ``(Gamma, F)`` to 1e-12, on the well-scaled base panel
    and on the ill-conditioned one."""
    panel_a, _, _, cfg, lam_max_a = base
    panel_c, _, _ = make_ill_panel()
    cfg_c = EstimationConfig(K=2, tol=1e-12, max_iter=300, penalize_intercept=pen_int)  # identity holds converged or not
    lam_max_c = si.lambda_max(panel_c, cfg_c)
    for panel, c, lmax in ((panel_a, replace(cfg, penalize_intercept=pen_int, max_iter=300), lam_max_a), (panel_c, cfg_c, lam_max_c)):
        for frac in (0.02, 0.1, 0.5):
            res = si.fit_sparse_ipca(panel, c, lam=frac * lmax)
            ref = si.objective_value(panel, res.Gamma, res.F, res.lam, panel.sigma_c, pen_int)
            assert res.obj_path[-1] == pytest.approx(ref, rel=1e-12)
            assert res.objective == res.obj_path[-1]
            assert res.obj_path[0] >= res.obj_path[-1]


def test_conditioning_of_the_gamma_step_on_an_ill_scaled_panel(backend):
    """Instrument columns with scales 1e-6..1e-4 next to the constant column (``make_ill_panel``,
    N=200, T=120, L=40, K=2). At the returned factors the Gamma-step Gram has condition number ~1e13
    in the original units and ~3e4..6e4 in the standardised coordinates (ratio ~1e8; the residual 1e4
    comes from the factor structure shared across topics, which no diagonal scaling removes).

    Numbers recorded on this panel (numba and numpy agree):

    * KKT residual of the Gamma-step at the returned solution, standardised coordinates, relative
      to ``||b~||`` with ``tol = 1e-12``: 8e-8..1e-7 (new solver). The previous original-unit solver
      gave the same 8e-8..1e-7 here: the residual at the returned pair is set by the *outer* ARLS
      tolerance, not by the coordinates (with the production ``tol = 1e-8`` it is ~2e-5 for both).
    * The Gamma-step itself, restarted from the returned solution with the production inner budget
      (``inner_max_iter = 200``, ``inner_tol = 1e-10``): standardised coordinates converge in 40-60
      sweeps with KKT~ ~1e-10 (entries of ``Gamma~`` are ~0.1-0.3, so the absolute tolerance is a
      relative one); original units do not converge (entries ~7e4 still move by ~3e-2 per sweep and
      need ~260 sweeps to reach an absolute 1e-10, i.e. ~1e-15 relative). Before the
      reparametrisation every penalised sweep of the fit ran this non-converging solve
      (``inner_iters`` 9800 vs 5335 for the same fit).
    * Unit invariance on this panel (column 5 x 1e6): old solver ``lambda_max`` 8.8e-2, ``F`` and
      ``Gamma`` 4.5e-2 relative change, fitted values 5e-8; new solver 7e-16 / 5e-15 / 8e-15 / 5e-15.
    * Full-size study panel (T=219, N=500, p=121, K=3, ``sigma_c`` 1.5e-7..1.5e-5, production
      config): cond(G) 2e14 vs cond(G~) 5e4; the Gamma-step from the returned solution needs ~3800
      sweeps in original units and ~1500 standardised to reach ``inner_tol`` (neither within 200:
      the correlated covariance instruments keep cond(G~) at 5e4, see the open point on
      ``inner_max_iter``); after 200 sweeps KKT~ is 6e-8 in both. Unit invariance there: old
      solver ``lambda_max`` 1.7e-1, ``F``/``Gamma`` 2.0e-1, fitted 1.4e-5; new 2e-16 / 1e-14 /
      3e-13 / 3e-14.
    """
    panel, _, _ = make_ill_panel()
    assert panel.sigma_c[1:].min() > 5e-7 and panel.sigma_c[1:].max() < 5e-4
    K = 2
    cfg = EstimationConfig(K=K, tol=1e-12, max_iter=2000)
    lam = 0.05 * si.lambda_max(panel, cfg)
    res = si.fit_sparse_ipca(panel, cfg, lam=lam)
    assert res.converged and 0 < res.n_selected < panel.L
    assert np.abs(res.Gamma[1:][res.selected]).max() > 1e3  # original-unit Gamma has the inverse scale of the columns
    mom = panel.moments()
    sigma = panel.sigma_c
    mom_t = si.standardized_moments(mom, sigma)
    G, b = si.gram_from_moments(mom.S, mom.V, res.F)
    G_t, b_t = si.gram_from_moments(mom_t.S, mom_t.V, res.F)
    cond, cond_t = np.linalg.cond(G), np.linalg.cond(G_t)
    assert cond / cond_t > 1e4, (cond, cond_t)
    pen = si.penalty_vector(lam, panel.n_obs, sigma)
    pen_t = si.standardized_penalties(lam, panel.n_obs, panel.p)
    x = res.Gamma.ravel()
    x_t = si.to_standardized(res.Gamma, sigma).ravel()
    kkt_t = gl.kkt_violation(G_t, b_t, x_t, K, pen_t) / np.linalg.norm(b_t)
    assert kkt_t < 1e-6, kkt_t
    # the Gamma-step restarted from the returned solution with the production inner budget
    x_s, info_s = group_lasso_gram(G_t, b_t, K, pen_t, x0=x_t, max_iter=200, tol=1e-10)
    assert info_s["converged"] and info_s["n_iter"] < 200
    assert gl.kkt_violation(G_t, b_t, x_s, K, pen_t) / np.linalg.norm(b_t) < 1e-9
    x_o, info_o = group_lasso_gram(G, b, K, pen, x0=x, max_iter=200, tol=1e-10)
    assert not info_o["converged"] and info_o["max_change"] > 1e-3  # the original-unit solve stalls above tol
    # both solves agree on the selected set and on the standardised solution (the stalled one is
    # already at its optimum in relative terms: the missing 1e-10 absolute is rounding of ~1e5 entries)
    sel_o = np.linalg.norm(x_o.reshape(panel.p, K), axis=1)[1:] > 0
    sel_s = np.linalg.norm(x_s.reshape(panel.p, K), axis=1)[1:] > 0
    np.testing.assert_array_equal(sel_o, sel_s)
    np.testing.assert_array_equal(sel_s, res.selected)
    assert_close(si.to_standardized(x_o.reshape(panel.p, K), sigma).ravel(), x_s, rtol=1e-6)
    # objective identity and unit invariance also hold on this panel
    assert res.objective == pytest.approx(si.objective_value(panel, res.Gamma, res.F, lam, sigma), rel=1e-12)
    col, factor = 5, 1e6
    rb = si.fit_sparse_ipca(rescaled_panel(panel, col, factor), cfg, lam=lam)
    np.testing.assert_array_equal(rb.selected, res.selected)
    assert rb.objective == pytest.approx(res.objective, rel=1e-9)
    Gb = rb.Gamma.copy()
    Gb[col] *= factor
    assert_close(Gb, res.Gamma)
    assert_close(rb.F, res.F)
