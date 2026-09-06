"""Tests for narrative_ipca.wrapup (BKS step 3, Section 6.1 Eq. 10-12, Section 6.3).

Every test checks a mathematical property -- a round trip, an identity, an
invariance or a hand-computed example -- not only shapes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from narrative_ipca.config import EvaluationConfig
from narrative_ipca.types import ShockPanel, SparseIPCAResult
from narrative_ipca.wrapup import (
    impact_vectors,
    project_observable,
    recover_A,
    retrieval_scores,
    state_variables,
    term_impact,
    wrap_up,
)

RCOND = 1e-12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _spd(rng: np.random.Generator, K: int) -> np.ndarray:
    """Random symmetric positive definite K x K matrix."""
    B = rng.standard_normal((K, K))
    return B @ B.T + K * np.eye(K)


def _gamma_tilde_from(A: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Eq. 5: Gamma_tilde = A (A'A)^-1 Sigma^-1 (dense reference, full rank)."""
    return A @ np.linalg.inv(A.T @ A) @ np.linalg.inv(Sigma)


def _month_end_trading_days(T: int) -> pd.DatetimeIndex:
    """Last business day of the first T calendar months from 2005-01."""
    cal = pd.bdate_range("2005-01-03", periods=T * 23 + 23)
    last = cal.to_series().groupby(cal.to_period("M")).last()
    return pd.DatetimeIndex(last.to_numpy()[:T])


def _shock_panel(z: np.ndarray, topics: list[str], window: int = 5) -> ShockPanel:
    idx = pd.bdate_range("2010-01-04", periods=z.shape[0])
    return ShockPanel(z=pd.DataFrame(z, index=idx, columns=topics), window=window)


def _make_fit(
    rng: np.random.Generator,
    L: int = 12,
    K: int = 3,
    T: int = 120,
    n_zero_rows: int = 0,
    F_rank: int | None = None,
    meta: dict | None = None,
) -> SparseIPCAResult:
    """A synthetic SparseIPCAResult with internally consistent mu_f / Sigma_ff."""
    Gamma = rng.standard_normal((L + 1, K))
    if n_zero_rows:
        Gamma[1 : 1 + n_zero_rows] = 0.0
    F = rng.standard_normal((T, K)) * 0.05 + 0.01
    if F_rank is not None and F_rank < K:
        # factors living in an F_rank-dimensional subspace -> Sigma_ff singular
        basis = np.linalg.qr(rng.standard_normal((K, F_rank)))[0]
        F = (F @ basis) @ basis.T
    populated = np.ones(T, dtype=bool)
    populated[::17] = False  # a few empty periods carry zeros, as the estimator does
    F[~populated] = 0.0
    Fp = F[populated]
    mu_f = Fp.mean(axis=0)
    Sigma_ff = np.cov(Fp, rowvar=False, ddof=1)
    norms = np.linalg.norm(Gamma, axis=1)
    periods = _month_end_trading_days(T)
    return SparseIPCAResult(
        Gamma=Gamma,
        F=F,
        mu_f=mu_f,
        Sigma_ff=np.atleast_2d(Sigma_ff),
        lam=0.1,
        K=K,
        objective=1.0,
        obj_path=[1.0],
        n_iter=1,
        converged=True,
        total_r2=0.1,
        pred_r2=0.01,
        gamma_norms=norms,
        selected=norms[1:] > 0,
        instrument_names=["const"] + [f"topic_{j}" for j in range(L)],
        periods=periods,
        n_obs=T * 50,
        populated=populated,
        meta=meta or {},
    )


# ---------------------------------------------------------------------------
# recover_A
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L,K", [(10, 3), (25, 1), (7, 5)])
def test_recover_A_round_trip(L: int, K: int) -> None:
    """A -> Gamma_tilde = A (A'A)^-1 Sigma^-1 -> recover_A gives A back."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((L, K))
    Sigma = _spd(rng, K)
    Gt = _gamma_tilde_from(A, Sigma)
    A_hat, flag = recover_A(Gt, Sigma, RCOND)
    assert A_hat.shape == (L, K)
    assert flag is False
    np.testing.assert_allclose(A_hat, A, rtol=1e-9, atol=1e-10)
    # and the forward direction reproduces Gamma_tilde from the recovered A
    np.testing.assert_allclose(_gamma_tilde_from(A_hat, Sigma), Gt, rtol=1e-9, atol=1e-10)


def test_recover_A_round_trip_with_zero_rows() -> None:
    """Dropped narratives (zero rows of Gamma_tilde) give zero rows of A and no flag when >= K survive."""
    rng = np.random.default_rng(2)
    L, K = 15, 3
    A = rng.standard_normal((L, K))
    A[[1, 4, 9]] = 0.0
    Sigma = _spd(rng, K)
    Gt = _gamma_tilde_from(A, Sigma)
    assert np.all(Gt[[1, 4, 9]] == 0.0)
    A_hat, flag = recover_A(Gt, Sigma, RCOND)
    assert flag is False
    np.testing.assert_allclose(A_hat, A, rtol=1e-9, atol=1e-10)
    assert np.all(A_hat[[1, 4, 9]] == 0.0)


def test_recover_A_rank_deficient_when_fewer_than_K_rows() -> None:
    """Only K-1 non-zero rows -> rank_deficient True, no exception, zero rows preserved."""
    rng = np.random.default_rng(3)
    L, K = 10, 3
    Gt = np.zeros((L, K))
    Gt[[2, 5]] = rng.standard_normal((2, K))  # K - 1 = 2 non-zero rows
    Sigma = _spd(rng, K)
    A_hat, flag = recover_A(Gt, Sigma, RCOND)
    assert flag is True
    assert A_hat.shape == (L, K)
    assert np.all(np.isfinite(A_hat))
    zero_rows = np.setdiff1d(np.arange(L), [2, 5])
    assert np.all(A_hat[zero_rows] == 0.0)
    # the minimum-norm A keeps the rank and the column space of Gamma_tilde
    assert np.linalg.matrix_rank(A_hat) == K - 1 == np.linalg.matrix_rank(Gt)
    proj = Gt @ np.linalg.pinv(Gt)  # orthogonal projector on col(Gt)
    np.testing.assert_allclose(proj @ A_hat, A_hat, rtol=1e-9, atol=1e-12)
    # the round trip A -> Gamma_tilde is exact when Sigma = I (pinv does not distribute over
    # Sigma^-1 (Gt'Gt)^+ Sigma^-1 otherwise; that is what the flag warns about)
    A_id, flag_id = recover_A(Gt, np.eye(K), RCOND)
    assert flag_id is True
    Gt_back = A_id @ np.linalg.pinv(A_id.T @ A_id, rcond=RCOND)
    np.testing.assert_allclose(Gt_back, Gt, rtol=1e-9, atol=1e-12)


def test_recover_A_rank_deficient_when_rows_collinear() -> None:
    """K non-zero rows that are linearly dependent still flag rank deficiency."""
    rng = np.random.default_rng(4)
    L, K = 8, 3
    Gt = np.zeros((L, K))
    row = rng.standard_normal(K)
    Gt[0], Gt[3], Gt[6] = row, 2.0 * row, -0.5 * row  # 3 non-zero rows, rank 1
    _, flag = recover_A(Gt, _spd(rng, K), RCOND)
    assert flag is True


def test_recover_A_rank_deficient_when_sigma_singular() -> None:
    """Singular Sigma_ff (factors in a K-1 subspace) flags rank deficiency and uses the pseudo-inverse."""
    rng = np.random.default_rng(5)
    L, K = 10, 3
    Gt = rng.standard_normal((L, K))
    q = np.linalg.qr(rng.standard_normal((K, K - 1)))[0]
    Sigma = q @ _spd(rng, K - 1) @ q.T  # rank K-1
    A_hat, flag = recover_A(Gt, Sigma, RCOND)
    assert flag is True
    assert np.all(np.isfinite(A_hat))
    # the pseudo-inverse of Sigma annihilates its null space, so A's rows lie in range(Sigma)
    null = np.linalg.svd(Sigma)[2][-1]
    np.testing.assert_allclose(A_hat @ null, 0.0, atol=1e-10)


def test_recover_A_all_zero_does_not_raise() -> None:
    A_hat, flag = recover_A(np.zeros((6, 2)), np.eye(2), RCOND)
    assert flag is True
    assert np.all(A_hat == 0.0)


def test_recover_A_rejects_bad_shapes_and_nonfinite() -> None:
    with pytest.raises(ValueError):
        recover_A(np.ones((5, 2)), np.eye(3), RCOND)
    G = np.ones((5, 2))
    G[0, 0] = np.nan
    with pytest.raises(ValueError):
        recover_A(G, np.eye(2), RCOND)


# ---------------------------------------------------------------------------
# state_variables
# ---------------------------------------------------------------------------
def test_state_variables_exact_without_noise() -> None:
    """z = A x exactly -> x recovered to machine precision."""
    rng = np.random.default_rng(6)
    L, K, n = 20, 3, 300
    A = rng.standard_normal((L, K))
    x = rng.standard_normal((n, K))
    z = x @ A.T
    states = state_variables(A, _shock_panel(z, [f"t{j}" for j in range(L)]), RCOND)
    assert list(states.columns) == ["x1", "x2", "x3"]
    np.testing.assert_allclose(states.to_numpy(), x, rtol=1e-10, atol=1e-10)


def test_state_variables_recovers_states_with_noise() -> None:
    """z = A x + eta with small eta -> corr(x_hat_k, x_k) > 0.99 for every k."""
    rng = np.random.default_rng(7)
    L, K, n = 30, 3, 2000
    A = rng.standard_normal((L, K))
    x = rng.standard_normal((n, K))
    z = x @ A.T + 0.3 * rng.standard_normal((n, L))
    states = state_variables(A, _shock_panel(z, [f"t{j}" for j in range(L)]), RCOND)
    for k in range(K):
        c = np.corrcoef(states.iloc[:, k].to_numpy(), x[:, k])[0, 1]
        assert c > 0.99, f"state {k}: corr {c:.4f}"
    # least-squares property: residual z - A x_hat is orthogonal to the columns of A (KKT)
    resid = z - states.to_numpy() @ A.T
    np.testing.assert_allclose(resid @ A, 0.0, atol=1e-8)


def test_state_variables_nan_rows_propagate() -> None:
    rng = np.random.default_rng(8)
    L, K, n = 6, 2, 40
    A = rng.standard_normal((L, K))
    z = rng.standard_normal((n, L))
    z[:5] = np.nan  # the first `window` days
    z[17, 2] = np.nan  # a single gap
    states = state_variables(A, _shock_panel(z, [f"t{j}" for j in range(L)]), RCOND)
    assert states.iloc[:5].isna().all().all()
    assert states.iloc[17].isna().all()
    finite_rows = np.setdiff1d(np.arange(n), np.r_[np.arange(5), 17])
    assert np.isfinite(states.iloc[finite_rows].to_numpy()).all()
    expected = z[finite_rows] @ A @ np.linalg.inv(A.T @ A)
    np.testing.assert_allclose(states.iloc[finite_rows].to_numpy(), expected, rtol=1e-10, atol=1e-12)
    assert states.index.equals(pd.bdate_range("2010-01-04", periods=n))


def test_state_variables_column_mismatch_raises() -> None:
    A = np.ones((4, 2))
    with pytest.raises(ValueError):
        state_variables(A, _shock_panel(np.ones((3, 5)), list("abcde")), RCOND)


# ---------------------------------------------------------------------------
# impact_vectors
# ---------------------------------------------------------------------------
def test_impact_vectors_identity_on_column_space() -> None:
    """For z = A x0 (in the column space of A): I_{z->MVE}' z = b_mve' x0 (Eq. 10-11)."""
    rng = np.random.default_rng(9)
    L, K = 14, 3
    A = rng.standard_normal((L, K))
    b = rng.standard_normal(K)
    I_zx, I_zmve = impact_vectors(A, b, RCOND)
    assert I_zx.shape == (L, K) and I_zmve.shape == (L,)
    for _ in range(5):
        x0 = rng.standard_normal(K)
        z = A @ x0
        assert I_zmve @ z == pytest.approx(b @ x0, rel=1e-10, abs=1e-12)
        np.testing.assert_allclose(I_zx.T @ z, x0, rtol=1e-10, atol=1e-12)
    # A' I_{z->x} = (A'A)(A'A)^-1 = I_K  and  I_{z->MVE} = I_{z->x} b
    np.testing.assert_allclose(A.T @ I_zx, np.eye(K), atol=1e-10)
    np.testing.assert_allclose(I_zmve, I_zx @ b, rtol=1e-12)
    # against a dense reference
    np.testing.assert_allclose(I_zx, A @ np.linalg.inv(A.T @ A), rtol=1e-10, atol=1e-12)


def test_impact_vectors_orthogonal_component_has_no_impact() -> None:
    """A shock orthogonal to the column space of A (pure eta) moves no state."""
    rng = np.random.default_rng(10)
    L, K = 10, 2
    A = rng.standard_normal((L, K))
    I_zx, I_zmve = impact_vectors(A, rng.standard_normal(K), RCOND)
    eta = rng.standard_normal(L)
    eta -= A @ np.linalg.solve(A.T @ A, A.T @ eta)  # project out col(A)
    np.testing.assert_allclose(I_zx.T @ eta, 0.0, atol=1e-10)
    assert abs(I_zmve @ eta) < 1e-10


def test_impact_vectors_shape_errors() -> None:
    with pytest.raises(ValueError):
        impact_vectors(np.ones((5, 2)), np.ones(3), RCOND)


# ---------------------------------------------------------------------------
# project_observable
# ---------------------------------------------------------------------------
def test_project_observable_recovers_known_weights() -> None:
    rng = np.random.default_rng(11)
    T, K = 200, 3
    periods = pd.bdate_range("2000-01-31", periods=T)
    F = pd.DataFrame(rng.standard_normal((T, K)) * 0.05, index=periods, columns=["f1", "f2", "f3"])
    b_true = np.array([0.8, -0.3, 1.5])
    a_true = 0.002
    exact = pd.Series(a_true + F.to_numpy() @ b_true, index=periods)
    b_hat, r2 = project_observable(F, exact)
    np.testing.assert_allclose(b_hat, b_true, rtol=1e-10, atol=1e-12)
    assert r2 == pytest.approx(1.0, abs=1e-12)

    noisy = exact + 0.005 * rng.standard_normal(T)
    b_hat, r2 = project_observable(F, noisy)
    np.testing.assert_allclose(b_hat, b_true, atol=0.05)
    assert 0.8 < r2 < 1.0
    # brute-force OLS reference with intercept
    X = np.column_stack([np.ones(T), F.to_numpy()])
    ref = np.linalg.solve(X.T @ X, X.T @ noisy.to_numpy())
    np.testing.assert_allclose(b_hat, ref[1:], rtol=1e-10)
    resid = noisy.to_numpy() - X @ ref
    r2_ref = 1 - resid @ resid / ((noisy - noisy.mean()) ** 2).sum()
    assert r2 == pytest.approx(r2_ref, rel=1e-12)


def test_project_observable_aligns_on_common_index_and_drops_nan() -> None:
    rng = np.random.default_rng(12)
    T, K = 100, 2
    periods = pd.bdate_range("2000-01-31", periods=T)
    F = pd.DataFrame(rng.standard_normal((T, K)), index=periods, columns=["f1", "f2"])
    b_true = np.array([1.0, -2.0])
    target = pd.Series(0.5 + F.to_numpy() @ b_true, index=periods)
    partial = target.iloc[30:].copy()
    partial.iloc[3] = np.nan
    extra = pd.concat([partial, pd.Series([1.0, 2.0], index=pd.bdate_range("2030-01-01", periods=2))])
    b_hat, r2 = project_observable(F, extra)
    np.testing.assert_allclose(b_hat, b_true, rtol=1e-10, atol=1e-12)
    assert r2 == pytest.approx(1.0, abs=1e-12)


def test_project_observable_period_matching() -> None:
    """Month-end-stamped target pairs with last-trading-day-stamped factors when period is given."""
    rng = np.random.default_rng(13)
    T, K = 60, 2
    trading_ends = _month_end_trading_days(T)
    F = pd.DataFrame(rng.standard_normal((T, K)), index=trading_ends, columns=["f1", "f2"])
    b_true = np.array([0.3, 0.7])
    month_ends = trading_ends.to_period("M").to_timestamp(how="end").normalize()
    target = pd.Series(F.to_numpy() @ b_true, index=month_ends)
    b_hat, r2 = project_observable(F, target, period="M")
    np.testing.assert_allclose(b_hat, b_true, rtol=1e-10, atol=1e-12)
    assert r2 == pytest.approx(1.0, abs=1e-12)


def test_project_observable_insufficient_overlap_raises() -> None:
    F = pd.DataFrame(np.ones((10, 3)), index=pd.bdate_range("2000-01-31", periods=10), columns=list("abc"))
    target = pd.Series([1.0, 2.0], index=pd.bdate_range("2030-01-01", periods=2))
    with pytest.raises(ValueError):
        project_observable(F, target)


# ---------------------------------------------------------------------------
# term_impact
# ---------------------------------------------------------------------------
def test_term_impact_hand_checked_2x3() -> None:
    """P = [[1, 0, 0], [0, .5, .5]] (rows sum to one), I_z = [a, b].

    P P' = diag(1, 0.5), (P P')^-1 = diag(1, 2), so I_w = P' diag(1, 2) I_z
    = P' [a, 2b] = [a, b, b].
    """
    phi = pd.DataFrame([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]], index=["t0", "t1"], columns=["w0", "w1", "w2"])
    a, b = 0.7, -1.3
    I_w = term_impact(phi, np.array([a, b]), RCOND)
    assert list(I_w.index) == ["w0", "w1", "w2"]
    np.testing.assert_allclose(I_w.to_numpy(), [a, b, b], rtol=1e-12)
    # the term-level vector reproduces the narrative-level impact for any dw:
    # z(dw) = (P P')^-1 P dw = [dw0, dw1 + dw2]  ->  I_z' z = a dw0 + b (dw1 + dw2) = I_w' dw
    dw = np.array([0.2, -0.4, 0.9])
    z = np.array([dw[0], dw[1] + dw[2]])
    assert I_w.to_numpy() @ dw == pytest.approx(a * z[0] + b * z[1], rel=1e-12)


def test_term_impact_identity_random() -> None:
    """I_w' dw == I_z' z(dw) with z(dw) = (P P')^-1 P dw for random P and dw (Eq. 12)."""
    rng = np.random.default_rng(14)
    L, V = 6, 40
    P = rng.random((L, V))
    P /= P.sum(axis=1, keepdims=True)
    phi = pd.DataFrame(P, index=[f"t{j}" for j in range(L)], columns=[f"w{v}" for v in range(V)])
    I_z = rng.standard_normal(L)
    I_w = term_impact(phi, I_z, RCOND)
    assert I_w.shape == (V,)
    # dense reference in BKS orientation: Phi = P' (V x L), I_w = Phi (Phi'Phi)^-1 I_z
    Phi = P.T
    np.testing.assert_allclose(I_w.to_numpy(), Phi @ np.linalg.inv(Phi.T @ Phi) @ I_z, rtol=1e-9, atol=1e-12)
    for _ in range(5):
        dw = rng.standard_normal(V)
        z = np.linalg.solve(P @ P.T, P @ dw)
        assert I_w.to_numpy() @ dw == pytest.approx(I_z @ z, rel=1e-9, abs=1e-12)


def test_term_impact_shape_mismatch_raises() -> None:
    phi = pd.DataFrame(np.ones((3, 4)) / 4)
    with pytest.raises(ValueError):
        term_impact(phi, np.ones(2), RCOND)


# ---------------------------------------------------------------------------
# wrap_up
# ---------------------------------------------------------------------------
def test_wrap_up_fills_every_field_and_aligns_columns() -> None:
    rng = np.random.default_rng(15)
    L, K, T = 12, 3, 120
    fit = _make_fit(rng, L=L, K=K, T=T)
    topics = fit.instrument_names[1:]
    n_days = 500
    z = rng.standard_normal((n_days, L))
    z[:5] = np.nan
    shocks = _shock_panel(z, topics)

    # observable = a + F b + noise on populated periods, indexed by the fit's periods
    Fp = fit.F[fit.populated]
    b_obs_true = np.array([1.0, 0.5, -0.25])
    obs = pd.Series(0.001 + Fp @ b_obs_true + 0.002 * rng.standard_normal(Fp.shape[0]), index=fit.periods[fit.populated])

    V = 30
    P = rng.random((L, V))
    P /= P.sum(axis=1, keepdims=True)
    phi = pd.DataFrame(P, index=topics, columns=[f"w{v}" for v in range(V)])

    res = wrap_up(fit, shocks, EvaluationConfig(), observables={"mkt": obs}, phi=phi)

    # A and the impact matrices: labels and equations
    assert list(res.A.index) == topics and list(res.A.columns) == ["f1", "f2", "f3"]
    A_ref, flag = recover_A(fit.Gamma_tilde, fit.Sigma_ff, 1e-12)
    np.testing.assert_allclose(res.A.to_numpy(), A_ref)
    assert res.rank_deficient is False and flag is False
    assert list(res.impact_z_to_x.index) == topics and list(res.impact_z_to_x.columns) == ["f1", "f2", "f3"]
    np.testing.assert_allclose(res.impact_z_to_x.to_numpy(), A_ref @ np.linalg.inv(A_ref.T @ A_ref), rtol=1e-9)
    np.testing.assert_allclose(res.b_mve, np.linalg.solve(fit.Sigma_ff, fit.mu_f), rtol=1e-9)
    np.testing.assert_allclose(res.impact_z_to_mve.to_numpy(), res.impact_z_to_x.to_numpy() @ res.b_mve, rtol=1e-12)
    assert list(res.impact_z_to_mve.index) == topics

    # states / x_mve on the shock dates
    assert res.states.index.equals(shocks.z.index) and res.x_mve.index.equals(shocks.z.index)
    assert res.states.shape == (n_days, K) and list(res.states.columns) == ["x1", "x2", "x3"]
    assert res.states.iloc[:5].isna().all().all() and res.x_mve.iloc[:5].isna().all()
    np.testing.assert_allclose(res.x_mve.to_numpy()[5:], res.states.to_numpy()[5:] @ res.b_mve, rtol=1e-12)
    # x_mve equals I_{z->MVE}' z_tau day by day (Eq. 11)
    np.testing.assert_allclose(res.x_mve.to_numpy()[5:], z[5:] @ res.impact_z_to_mve.to_numpy(), rtol=1e-9)

    # observable projection and impact vector
    assert set(res.impact_z_to_obs) == {"mkt"} and set(res.obs_projection) == {"mkt"}
    proj = res.obs_projection["mkt"]
    np.testing.assert_allclose(proj["b_obs"], b_obs_true, atol=0.05)
    assert 0.9 < proj["r2"] <= 1.0
    assert proj["n_obs"] == int(fit.populated.sum())
    np.testing.assert_allclose(
        res.impact_z_to_obs["mkt"].to_numpy(), res.impact_z_to_x.to_numpy() @ proj["b_obs"], rtol=1e-12
    )
    assert list(res.impact_z_to_obs["mkt"].index) == topics

    # term level
    assert res.impact_w_to_mve is not None and list(res.impact_w_to_mve.index) == list(phi.columns)
    np.testing.assert_allclose(
        res.impact_w_to_mve.to_numpy(), term_impact(phi, res.impact_z_to_mve.to_numpy(), 1e-12).to_numpy()
    )
    assert set(res.meta["impact_w_to_obs"]) == {"mkt"}
    assert res.meta["n_selected"] == L and res.meta["n_days_finite"] == n_days - 5

    # column permutation of the shocks and row permutation of phi must not change anything
    perm = rng.permutation(L)
    shocks_perm = ShockPanel(z=shocks.z.iloc[:, perm], window=shocks.window)
    res_perm = wrap_up(fit, shocks_perm, EvaluationConfig(), observables={"mkt": obs}, phi=phi.iloc[perm])
    pd.testing.assert_frame_equal(res_perm.states, res.states)
    pd.testing.assert_series_equal(res_perm.impact_z_to_mve, res.impact_z_to_mve)
    pd.testing.assert_series_equal(res_perm.impact_w_to_mve, res.impact_w_to_mve)


def test_wrap_up_missing_topic_column_raises() -> None:
    rng = np.random.default_rng(16)
    fit = _make_fit(rng, L=5, K=2, T=50)
    shocks = _shock_panel(rng.standard_normal((30, 4)), fit.instrument_names[1:5])
    with pytest.raises(ValueError):
        wrap_up(fit, shocks, EvaluationConfig())


@pytest.mark.parametrize("orthogonal", [True, False])
def test_wrap_up_rotation_invariance(orthogonal: bool) -> None:
    """Gamma -> Gamma R, f -> R^-1 f leaves x_MVE and I_{z->MVE} unchanged; A -> A R, x -> R^-1 x."""
    rng = np.random.default_rng(17)
    L, K, T = 10, 3, 150
    fit = _make_fit(rng, L=L, K=K, T=T)
    topics = fit.instrument_names[1:]
    z = rng.standard_normal((400, L))
    z[:5] = np.nan
    shocks = _shock_panel(z, topics)
    cfg = EvaluationConfig()
    base = wrap_up(fit, shocks, cfg)

    if orthogonal:
        R = np.linalg.qr(rng.standard_normal((K, K)))[0]
        R[:, 0] *= -1.0  # include a sign flip
    else:
        R = rng.standard_normal((K, K)) + 3.0 * np.eye(K)
    rot = wrap_up(fit.rotate(R), shocks, cfg)

    # invariant objects
    pd.testing.assert_series_equal(rot.x_mve, base.x_mve, rtol=1e-8, atol=1e-10)
    pd.testing.assert_series_equal(rot.impact_z_to_mve, base.impact_z_to_mve, rtol=1e-8, atol=1e-10)
    assert rot.rank_deficient is False
    # covariant objects: A -> A R ; x -> R^-1 x, i.e. X -> X R^-T ; I_{z->x} -> I_{z->x} R^-T ; b -> R' b
    Rinv = np.linalg.inv(R)
    np.testing.assert_allclose(rot.A.to_numpy(), base.A.to_numpy() @ R, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(rot.states.to_numpy()[5:], base.states.to_numpy()[5:] @ Rinv.T, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(rot.impact_z_to_x.to_numpy(), base.impact_z_to_x.to_numpy() @ Rinv.T, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(rot.b_mve, R.T @ base.b_mve, rtol=1e-8, atol=1e-10)


def test_wrap_up_observable_impact_is_rotation_invariant() -> None:
    rng = np.random.default_rng(18)
    L, K, T = 8, 2, 100
    fit = _make_fit(rng, L=L, K=K, T=T)
    shocks = _shock_panel(rng.standard_normal((60, L)), fit.instrument_names[1:])
    Fp = fit.F[fit.populated]
    obs = pd.Series(Fp @ np.array([0.4, -0.9]) + 0.01 * rng.standard_normal(Fp.shape[0]), index=fit.periods[fit.populated])
    cfg = EvaluationConfig()
    base = wrap_up(fit, shocks, cfg, observables={"m": obs})
    Q = np.linalg.qr(rng.standard_normal((K, K)))[0]
    rot = wrap_up(fit.rotate(Q), shocks, cfg, observables={"m": obs})
    pd.testing.assert_series_equal(rot.impact_z_to_obs["m"], base.impact_z_to_obs["m"], rtol=1e-8, atol=1e-10)
    assert rot.obs_projection["m"]["r2"] == pytest.approx(base.obs_projection["m"]["r2"], rel=1e-10)


def test_wrap_up_rank_deficient_fit_sets_flag_and_completes() -> None:
    """Only K-1 narratives selected and factors in a K-1 subspace: flag set, no exception, finite output."""
    rng = np.random.default_rng(19)
    L, K, T = 9, 3, 80
    fit = _make_fit(rng, L=L, K=K, T=T, n_zero_rows=L - (K - 1), F_rank=K - 1)
    assert fit.n_selected == K - 1
    shocks = _shock_panel(rng.standard_normal((50, L)), fit.instrument_names[1:])
    res = wrap_up(fit, shocks, EvaluationConfig())
    assert res.rank_deficient is True
    assert np.isfinite(res.A.to_numpy()).all()
    assert np.isfinite(res.states.to_numpy()).all()
    assert np.isfinite(res.x_mve.to_numpy()).all()
    # dropped narratives have zero loadings and zero impact
    dropped = ~fit.selected
    assert np.all(res.A.to_numpy()[dropped] == 0.0)
    assert np.all(res.impact_z_to_mve.to_numpy()[dropped] == 0.0)


def test_wrap_up_uses_period_from_fit_meta() -> None:
    rng = np.random.default_rng(20)
    L, K, T = 6, 2, 60
    fit = _make_fit(rng, L=L, K=K, T=T, meta={"period": "M"})
    # the fit's periods are last trading days; stamp the observable at calendar month ends
    Fp = fit.F[fit.populated]
    month_ends = fit.periods[fit.populated].to_period("M").to_timestamp(how="end").normalize()
    assert month_ends.is_unique
    assert len(month_ends.intersection(fit.periods)) < len(month_ends)  # exact matching would lose rows
    obs = pd.Series(Fp @ np.array([1.0, 2.0]), index=month_ends)
    shocks = _shock_panel(rng.standard_normal((30, L)), fit.instrument_names[1:])
    res = wrap_up(fit, shocks, EvaluationConfig(), observables={"m": obs})
    np.testing.assert_allclose(res.obs_projection["m"]["b_obs"], [1.0, 2.0], rtol=1e-9)
    assert res.obs_projection["m"]["n_obs"] == int(fit.populated.sum())


# ---------------------------------------------------------------------------
# retrieval_scores
# ---------------------------------------------------------------------------
def test_retrieval_scores_inner_product_per_row_and_nan_safe() -> None:
    rng = np.random.default_rng(21)
    L, n = 5, 12
    topics = [f"t{j}" for j in range(L)]
    I_z = pd.Series(rng.standard_normal(L), index=topics, name="impact_z_to_mve")
    z = rng.standard_normal((n, L))
    z[2] = np.nan
    z[7, 1] = np.nan
    frame = pd.DataFrame(z, index=[f"article_{i}" for i in range(n)], columns=topics)
    scores = retrieval_scores(I_z, frame)
    assert scores.index.equals(frame.index) and scores.name == "impact_z_to_mve"
    assert np.isnan(scores.iloc[2]) and np.isnan(scores.iloc[7])
    ok = np.setdiff1d(np.arange(n), [2, 7])
    np.testing.assert_allclose(scores.to_numpy()[ok], z[ok] @ I_z.to_numpy(), rtol=1e-12)
    # brute force per row
    for i in ok:
        assert scores.iloc[i] == pytest.approx(sum(z[i, j] * I_z.iloc[j] for j in range(L)), rel=1e-12)


def test_retrieval_scores_aligns_columns_by_name() -> None:
    rng = np.random.default_rng(22)
    topics = ["a", "b", "c"]
    I_z = pd.Series([1.0, -2.0, 0.5], index=topics)
    z = pd.DataFrame(rng.standard_normal((4, 4)), columns=["c", "extra", "a", "b"])
    scores = retrieval_scores(I_z, z)
    expected = z["a"] * 1.0 + z["b"] * -2.0 + z["c"] * 0.5
    np.testing.assert_allclose(scores.to_numpy(), expected.to_numpy(), rtol=1e-12)
    with pytest.raises(ValueError):
        retrieval_scores(I_z, z[["a", "b"]])


def test_retrieval_scores_matches_x_mve_from_wrap_up() -> None:
    """Scoring the daily shocks with I_{z->MVE} reproduces x_MVE (Eq. 11 day by day)."""
    rng = np.random.default_rng(23)
    L, K = 7, 2
    fit = _make_fit(rng, L=L, K=K, T=90)
    z = rng.standard_normal((80, L))
    z[:5] = np.nan
    shocks = _shock_panel(z, fit.instrument_names[1:])
    res = wrap_up(fit, shocks, EvaluationConfig())
    scores = retrieval_scores(res.impact_z_to_mve, shocks.z)
    pd.testing.assert_series_equal(scores.rename("x_mve"), res.x_mve, rtol=1e-9, atol=1e-12)
