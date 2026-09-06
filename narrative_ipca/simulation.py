"""Simulation of the BKS data-generating process with known ground truth (DESIGN.md Part D).

The simulated world follows Figure 1 of Bybee, Kelly & Su (2023) (BKS):

* daily tradable factors ``f_tau`` (K x 1) and states ``x_tau = f_tau + nu_tau``
  with ``nu`` orthogonal to returns (BKS Section 1.2, ``nu_tau := x_tau - f_tau``);
* topic shocks ``z_tau = A x_tau + eta_tau`` (BKS Eq. 1) with a row-sparse
  ``A`` (L x K): ``n_relevant`` non-zero rows, ``n_placebo`` pure-noise rows,
  the rest zero rows;
* daily attention *levels* ``theta_tau`` built from the shocks (additive or
  softmax mapping, Part D step 4), which is what the pipeline receives;
* daily asset returns ``r_{i,tau} = beta_{i,t(tau)}' f_tau + eps_{i,tau}``
  (BKS Eq. 4) with slowly varying loadings and class-specific idiosyncratic
  volatility, on an unbalanced panel with random missing days.

Units of the ground truth
-------------------------
The pipeline never sees ``x``; it recovers shocks from the attention levels as
``theta_tau - MA_w(theta)`` and estimates everything in *attention units*.
To make the truth comparable with those estimates without an extra per-topic
conversion, ``SimulationTruth.A`` and ``SimulationTruth.z_daily`` are expressed
in attention units too: ``z_daily`` is the shock that enters ``theta``
additively (additive model) or its first-order equivalent (softmax model), and
``A`` is the matrix such that ``z_daily = A x + eta`` in those units. In the
notation of DESIGN.md Part D, ``A = diag(shock_scale) A_rel`` where ``A_rel``
holds the ``signal_strength * N(0, 1)`` draws (in *relative* units, i.e. as a
fraction of the topic's typical level) and ``shock_scale_l`` is ``m_l`` (the
topic's mean level; additive) or ``theta_bar_l = m_l / sum_j m_j`` (softmax).
The per-topic scale is reported in ``truth.meta["shock_scale"]``.

Relative shock scale
--------------------
DESIGN.md leaves the size of a daily attention shock relative to the topic's
level unspecified. It is fixed here by the module constant
:data:`SHOCK_RELATIVE_VOL` (= 0.12): the states enter the relative shock as
``kappa * A_rel x`` with ``kappa = SHOCK_RELATIVE_VOL / sqrt(tr Sigma_x)``,
``Sigma_x = (1 + nontradable_share) Sigma_ff_daily``, so that a relevant topic
at ``signal_strength = 1`` has *expected* signal variance
``SHOCK_RELATIVE_VOL^2`` (because ``E[a' Sigma_x a] = tr Sigma_x`` for
``a ~ N(0, I_K)``), and the topic noise has standard deviation
``SHOCK_RELATIVE_VOL * topic_noise_vol * s_l`` with ``s_l ~ U(0.5, 1.5)``. The
expected signal-to-noise ratio of a relevant topic is therefore
``1 / (topic_noise_vol * s_l)^2``, i.e. O(1) at the default
``topic_noise_vol = 1``. The value 0.12 keeps the additive levels positive in
all but roughly 0.04 % of topic-days at the default slow-component volatility
(``attention_slow_vol = 0.05`` with persistence 0.98 has stationary relative
std 0.25), below the :data:`CLIP_WARN_FRACTION` warning threshold.

Calendar and annualisation
--------------------------
The calendar is ``pd.bdate_range`` (about 21.7 business days per calendar
month), so the *realised* mean number of trading days per period
``d_bar = n_days / T`` is what converts period moments to daily ones:
daily factor moments are ``mu_f_period / d_bar`` and ``Sigma_ff_period / d_bar``
and the daily idiosyncratic std is ``idio_vol / sqrt(ann * d_bar)``, so that
the period idiosyncratic variance is exactly ``idio_vol^2 / ann`` with
``ann = periods_per_year`` (12 for monthly periods). Using the nominal
``days_per_year / periods_per_year = 21`` instead would put the realised
period moments 3.4 % off their population values.

The two null scenarios
----------------------
``topic_null`` (alias ``null``) sets ``signal_strength = 0`` and keeps the
priced factor structure of returns: every topic is noise, but the kernel
covariance of a noise topic with asset ``i`` is ``beta_i' G_{t,l}`` plus
idiosyncratic noise, with ``G_{t,l} = sum_tau w_tau f_tau z_{l,tau}`` a common
``K``-vector that is non-zero at order ``1/sqrt(n_eff)`` and persistent over
``t`` (the kernel half-life is 69 months). With ``L`` noise topics the
instruments therefore span ``beta`` and the estimator recovers the factor
premium from pure noise topics: selection above chance and a positive OOS
Sharpe ratio are the *correct* behaviour there, so the harness reports this
scenario instead of testing it. ``no_factor`` additionally removes the common
factor structure from returns (every loading is zero, returns are pure
idiosyncratic noise with the class volatilities): that is the chance-level
null against which selection and the OOS Sharpe ratio are tested.

Everything is drawn from a single ``numpy.random.default_rng(cfg.seed)`` stream
in a fixed order, so a config reproduces its data exactly. ``no_factor`` and
``topic_null`` consume the stream exactly as ``baseline`` does (zero loadings
are drawn and multiplied by zero), so for a given seed the three scenarios
share their calendar, factor draws, topic assignment and idiosyncratic noise.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .config import AssetClassSpec, SimulationConfig
from .types import AttentionData, ReturnsData, SimulatedData, SimulationTruth, annualized_sharpe

__all__ = ["simulate", "scenario_config", "SCENARIOS", "SCENARIO_ALIASES", "FAST_OVERRIDES", "SHOCK_RELATIVE_VOL"]

logger = logging.getLogger(__name__)

SHOCK_RELATIVE_VOL: float = 0.12
"""Expected std of a relevant topic's *relative* attention shock at ``signal_strength = 1``."""

CALENDAR_START: str = "2005-01-03"
"""First trading day of the simulated calendar (DESIGN.md Part D)."""

CLIP_FLOOR: float = 1e-6
"""Additive attention levels are clipped at this floor (Part D step 4)."""

CLIP_WARN_FRACTION: float = 1e-3
"""A warning is logged when clipping affects more than this fraction of entries."""

SCENARIOS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "topic_null": {"signal_strength": 0.0},
    "no_factor": {"signal_strength": 0.0, "beta_innov_sd": 0.0},
    "softmax": {"attention_model": "softmax"},
    "weak": {"signal_strength": 0.35, "mve_sharpe_annual": 0.6},
    "balanced": {"unbalanced_fraction": 0.0, "missing_day_fraction": 0.0},
}
"""Field overrides of the named scenarios (DESIGN.md Part D).

``no_factor`` additionally replaces every :class:`AssetClassSpec` by a copy
with zero mean loadings and zero loading dispersion (see
:func:`_no_factor_classes`); that transformation is not a field override.
"""

SCENARIO_ALIASES: dict[str, str] = {"null": "topic_null"}
"""Accepted alternative names: ``null`` is the former name of ``topic_null``."""

FAST_OVERRIDES: dict[str, Any] = {
    "n_assets": 150,
    "n_topics": 40,
    "n_relevant": 8,
    "n_placebo": 8,
    "n_years": 8,
}
"""Reduced sizes used by the harness's fast mode (``HarnessConfig.fast``)."""

_PERIODS_PER_YEAR: dict[str, float] = {
    "M": 12.0, "ME": 12.0, "MS": 12.0, "BM": 12.0, "BME": 12.0, "BMS": 12.0,
    "W": 52.0,
    "Q": 4.0, "QE": 4.0, "QS": 4.0, "BQ": 4.0, "BQE": 4.0, "BQS": 4.0,
    "Y": 1.0, "YE": 1.0, "YS": 1.0, "A": 1.0, "AS": 1.0,
}

_PERIOD_ALIAS_TO_PERIOD_FREQ: dict[str, str] = {
    "ME": "M", "MS": "M", "BM": "M", "BME": "M", "BMS": "M",
    "QE": "Q", "QS": "Q", "BQ": "Q", "BQE": "Q", "BQS": "Q",
    "YE": "Y", "YS": "Y", "A": "Y", "AS": "Y",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def scenario_config(name: str, base: SimulationConfig | None = None, fast: bool = False) -> SimulationConfig:
    """Return the :class:`SimulationConfig` of a named scenario (DESIGN.md Part D).

    Scenarios apply field overrides on top of ``base`` (default
    ``SimulationConfig()``):

    * ``baseline``: the defaults (``base`` unchanged);
    * ``topic_null`` (alias ``null``): ``signal_strength = 0`` (every topic
      is noise) with the priced factor structure of returns kept; this is
      the informative null the harness reports rather than tests (see the
      module docstring);
    * ``no_factor``: ``signal_strength = 0`` *and* no common factor
      structure in returns: every asset class is replaced by a copy with
      ``beta_mean`` all zero and ``beta_sd = 0``, and ``beta_innov_sd = 0``,
      so returns are pure idiosyncratic noise with the class volatilities;
      this is the chance-level null;
    * ``softmax``: attention on the simplex (``attention_model = "softmax"``);
    * ``weak``: ``signal_strength = 0.35``, ``mve_sharpe_annual = 0.6``;
    * ``balanced``: ``unbalanced_fraction = 0``, ``missing_day_fraction = 0``.

    A fast variant (``fast=True``, or a ``-fast`` / ``_fast`` suffix on the
    name, or the bare name ``"fast"`` for the fast baseline) first applies
    :data:`FAST_OVERRIDES` (150 assets, 40 topics, 8 relevant, 8 placebo,
    8 years) to ``base`` and then the scenario overrides. Any other size can
    be obtained by passing a ``base`` with the desired sizes. Matching of
    ``name`` is case-insensitive.
    """
    key = name.strip().lower().replace("-", "_")
    if key == "fast":
        key, fast = "baseline", True
    elif key.endswith("_fast"):
        key, fast = key[: -len("_fast")], True
    key = SCENARIO_ALIASES.get(key, key)
    if key not in SCENARIOS:
        raise ValueError(
            f"unknown scenario {name!r}; known: {sorted(SCENARIOS)} (aliases {sorted(SCENARIO_ALIASES)}; "
            "optionally with a '-fast' suffix)"
        )
    cfg = SimulationConfig() if base is None else base
    if fast:
        cfg = replace(cfg, **FAST_OVERRIDES)
    cfg = replace(cfg, **SCENARIOS[key])
    if key == "no_factor":
        cfg = replace(cfg, asset_classes=_no_factor_classes(cfg.asset_classes))
    return cfg


def _no_factor_classes(classes: tuple[AssetClassSpec, ...]) -> tuple[AssetClassSpec, ...]:
    """Copies of ``classes`` with ``beta_mean`` all zero and ``beta_sd = 0`` (the ``no_factor`` scenario).

    Names, shares and idiosyncratic volatilities are kept, so the returns of
    ``no_factor`` have the class volatilities of the base configuration.
    """
    return tuple(
        replace(spec, beta_mean=(0.0,) * max(1, len(tuple(spec.beta_mean))), beta_sd=0.0) for spec in classes
    )


def simulate(cfg: SimulationConfig, *, scenario: str | None = None) -> SimulatedData:
    """Draw one simulated data set with ground truth (DESIGN.md Part D, steps 1-7).

    Parameters
    ----------
    cfg:
        The data-generating process.
    scenario:
        Name recorded in ``truth.meta["scenario"]``. ``None`` infers the most
        specific named scenario whose overrides ``cfg`` satisfies.

    Returns
    -------
    SimulatedData
        ``attention`` (daily levels ``theta``, one column per topic id
        ``topic_001``...), ``returns`` (daily excess returns, ``NaN`` outside
        the universe, asset ids ``A0001``..., ``asset_meta`` with
        ``asset_class``, ``entry``, ``exit``), ``truth`` and ``config``.

    The steps, in the order the random stream is consumed:

    1. Calendar ``n_years * days_per_year`` business days from 2005-01-03,
       periods ``cfg.period`` of that calendar stamped at their last trading
       day. Daily factors ``f_tau ~ N(mu_f_period / d_bar, Sigma_ff_period / d_bar)``
       with ``d_bar = n_days / T`` the realised mean number of trading days
       per period (about 21.7 for monthly periods of a business-day
       calendar; ``truth.meta["d_bar"]``), optionally AR(1); ``f_period`` =
       per-period sums. ``ann = periods_per_year`` (12 for monthly periods)
       is the annualisation of the period moments.
    2. States ``x_tau = f_tau + nu_tau``, ``nu ~ N(0, nontradable_share * Sigma_ff_daily)``.
    3. Topic positions (a random permutation), loadings ``A``, topic noise
       ``eta``, shocks ``z``, placebo columns.
    4. Attention levels (additive or softmax).
    5. Asset classes, long-run and period loadings, daily returns with
       idiosyncratic daily std ``idio_vol / sqrt(ann * d_bar)`` (period
       idiosyncratic variance exactly ``idio_vol^2 / ann``).
    6. Entry/exit dates and missing days.
    7. Ground-truth objects.

    See the module docstring for the units of ``truth.A`` / ``truth.z_daily``.
    """
    rng = np.random.default_rng(cfg.seed)
    K, L, N = cfg.K, cfg.n_topics, cfg.n_assets

    # -- step 1: calendar, periods, factors ---------------------------------
    calendar = pd.bdate_range(CALENDAR_START, periods=cfg.n_years * cfg.days_per_year)
    period_id, period_ends = _period_index(calendar, cfg.period)
    n_days, T = len(calendar), len(period_ends)
    ann = _periods_per_year(cfg.period)
    if ann is None:  # unknown alias: use the realised average period length
        ann = float(cfg.days_per_year * T / n_days)
    d_bar = n_days / T  # realised mean trading days per period (module docstring: calendar)

    mu_p, Sigma_p = _factor_moments(cfg, ann)
    Sigma_d = Sigma_p / d_bar
    f_daily = _draw_factors(rng, n_days, mu_p / d_bar, Sigma_d, cfg.factor_ar1)
    f_period = np.zeros((T, K))
    np.add.at(f_period, period_id, f_daily)

    # -- step 2: states -----------------------------------------------------
    Sigma_nu = cfg.nontradable_share * Sigma_d
    nu = rng.standard_normal((n_days, K)) * np.sqrt(np.diag(Sigma_nu))
    x_daily = f_daily + nu
    Sigma_x = Sigma_d + Sigma_nu

    # -- step 3: topics -----------------------------------------------------
    perm = rng.permutation(L)
    relevant = np.zeros(L, dtype=bool)
    placebo = np.zeros(L, dtype=bool)
    relevant[perm[: cfg.n_relevant]] = True
    placebo[perm[cfg.n_relevant : cfg.n_relevant + cfg.n_placebo]] = True

    A_rel = np.zeros((L, K))
    kappa = SHOCK_RELATIVE_VOL / np.sqrt(np.trace(Sigma_x))
    A_rel[relevant] = cfg.signal_strength * kappa * rng.standard_normal((cfg.n_relevant, K))

    s_l = rng.uniform(0.5, 1.5, size=L)
    sd_eta = SHOCK_RELATIVE_VOL * cfg.topic_noise_vol * s_l
    eta = _draw_topic_noise(rng, n_days, sd_eta, cfg.topic_noise_ar1, iid_mask=placebo)
    zeta = x_daily @ A_rel.T + eta  # relative units (fraction of the topic level)
    if cfg.n_placebo > 0:
        # BKS App. C.2: each placebo is i.i.d. normal with the time-series std of a randomly
        # chosen *real* narrative (Part D step 3: a relevant one). With no relevant topics
        # (n_relevant = 0 is legal when signal_strength = 0) fall back to any non-placebo
        # topic; the pool of relevant topics is used whenever it is non-empty, so the
        # random stream of every configuration with n_relevant > 0 is unchanged.
        pool = np.flatnonzero(relevant) if cfg.n_relevant > 0 else np.flatnonzero(~placebo)
        if pool.size == 0:  # every topic is a placebo: nothing to match, keep the noise draw
            pool = np.flatnonzero(placebo)
        match = rng.choice(pool, size=cfg.n_placebo, replace=True)
        sd_match = zeta[:, match].std(axis=0, ddof=1)
        zeta[:, placebo] = rng.standard_normal((n_days, cfg.n_placebo)) * sd_match

    # -- step 4: attention levels -------------------------------------------
    w = np.exp(cfg.attention_level_dispersion * rng.standard_normal(L))
    m = cfg.attention_level_mean * w / w.sum()  # mean levels; sum_l m_l = attention_level_mean
    s_slow = _ar1(rng, (n_days, L), cfg.attention_persistence, cfg.attention_slow_vol)
    theta, shock_scale, clipped_fraction = _attention_levels(cfg.attention_model, m, s_slow, zeta)
    A_true = shock_scale[:, None] * A_rel
    z_daily = shock_scale[None, :] * zeta

    # -- step 5: assets -----------------------------------------------------
    class_counts = _class_counts(cfg.asset_classes, N)
    asset_class = np.concatenate(
        [np.full(c, spec.name, dtype=object) for spec, c in zip(cfg.asset_classes, class_counts)]
    )
    idio_vol = np.concatenate([np.full(c, spec.idio_vol_annual) for spec, c in zip(cfg.asset_classes, class_counts)])
    beta_bar = np.zeros((N, K))
    start = 0
    for spec, c in zip(cfg.asset_classes, class_counts):
        mean = _pad(np.asarray(spec.beta_mean, dtype=float), K, fill=0.0)
        beta_bar[start : start + c] = mean + spec.beta_sd * rng.standard_normal((c, K))
        start += c
    b_dev = _ar1(rng, (T, N, K), cfg.beta_ar1, cfg.beta_innov_sd)
    beta_full = beta_bar[None, :, :] + b_dev  # (T, N, K)

    sd_eps = idio_vol / np.sqrt(ann * d_bar)  # period idiosyncratic variance = idio_vol^2 / ann exactly
    eps = _draw_idiosyncratic(rng, (n_days, N), cfg.fat_tails_df) * sd_eps[None, :]
    r = np.empty((n_days, N))
    for t in range(T):
        days = np.flatnonzero(period_id == t)
        r[days] = f_daily[days] @ beta_full[t].T + eps[days]

    # -- step 6: unbalanced panel and missing days -------------------------
    observed = _entry_exit(rng, n_days, N, cfg.unbalanced_fraction)
    if cfg.missing_day_fraction > 0:
        observed &= rng.random((n_days, N)) >= cfg.missing_day_fraction
    r = np.where(observed, r, np.nan)

    in_universe = np.zeros((T, N), dtype=bool)
    np.logical_or.at(in_universe, period_id, observed)
    beta = np.where(in_universe[:, :, None], beta_full, np.nan)

    # -- step 7: ground truth ----------------------------------------------
    AtA_inv = np.linalg.pinv(A_true.T @ A_true)
    Gamma_tilde_true = A_true @ AtA_inv @ np.linalg.inv(Sigma_d)
    impact_true = A_true @ AtA_inv @ np.linalg.solve(Sigma_p, mu_p)
    sharpe_true = annualized_sharpe(mu_p, Sigma_p, annualization=ann)
    systematic_r2 = _systematic_r2(beta_bar, Sigma_p, idio_vol, ann, cfg.beta_ar1, cfg.beta_innov_sd)
    b_true = np.linalg.solve(Sigma_p, mu_p)
    mve_series = f_period @ b_true
    sharpe_realized = float(mve_series.mean() / mve_series.std(ddof=1) * np.sqrt(ann)) if T > 1 else float("nan")

    # -- assemble -------------------------------------------------------------
    topic_ids = [f"topic_{l + 1:0{max(3, len(str(L)))}d}" for l in range(L)]
    asset_ids = [f"A{i + 1:0{max(4, len(str(N)))}d}" for i in range(N)]
    factor_cols = [f"f{k + 1}" for k in range(K)]
    kinds = np.where(relevant, "relevant", np.where(placebo, "placebo", "noise"))
    topic_labels = {tid: f"{kind} ({tid})" for tid, kind in zip(topic_ids, kinds)}

    ever = observed.any(axis=0)
    first_day = np.argmax(observed, axis=0)
    last_day = n_days - 1 - np.argmax(observed[::-1], axis=0)
    entry = pd.DatetimeIndex(np.where(ever, calendar.values[first_day], np.datetime64("NaT")))
    exit_ = pd.DatetimeIndex(np.where(ever, calendar.values[last_day], np.datetime64("NaT")))
    asset_meta = pd.DataFrame(
        {"asset_class": asset_class, "entry": entry, "exit": exit_},
        index=pd.Index(asset_ids, name="asset"),
    )

    scenario_name = scenario if scenario is not None else _infer_scenario(cfg)
    if clipped_fraction > CLIP_WARN_FRACTION:
        logger.warning(
            "additive attention levels clipped at %.0e for %.3f%% of topic-days (> %.1f%%); "
            "lower attention_slow_vol / topic_noise_vol or use the softmax model",
            CLIP_FLOOR, 100 * clipped_fraction, 100 * CLIP_WARN_FRACTION,
        )
    logger.info(
        "simulated scenario=%s seed=%d: %d days, %d periods (%s), %d assets, %d topics "
        "(%d relevant, %d placebo), K=%d, true MVE Sharpe %.2f (realised %.2f), systematic R2 %.2f",
        scenario_name, cfg.seed, n_days, T, cfg.period, N, L, cfg.n_relevant, cfg.n_placebo, K,
        sharpe_true, sharpe_realized, systematic_r2,
    )

    truth = SimulationTruth(
        A=A_true,
        relevant=relevant,
        placebo=placebo,
        f_daily=pd.DataFrame(f_daily, index=calendar, columns=factor_cols),
        f_period=pd.DataFrame(f_period, index=period_ends, columns=factor_cols),
        x_daily=pd.DataFrame(x_daily, index=calendar, columns=factor_cols),
        z_daily=pd.DataFrame(z_daily, index=calendar, columns=topic_ids),
        beta=beta,
        periods=period_ends,
        mu_f_period=mu_p,
        Sigma_ff_period=Sigma_p,
        Sigma_ff_daily=Sigma_d,
        sharpe_mve_true=sharpe_true,
        Gamma_tilde_true=Gamma_tilde_true,
        impact_z_to_mve_true=impact_true,
        asset_class=asset_class,
        systematic_r2=systematic_r2,
        meta={
            "scenario": scenario_name,
            "seed": int(cfg.seed),
            "sharpe_mve_realized": sharpe_realized,
            "periods_per_year": float(ann),
            "d_bar": float(d_bar),
            "mean_days_per_period": float(n_days / T),
            "n_days": int(n_days),
            "n_periods": int(T),
            "shock_relative_vol": SHOCK_RELATIVE_VOL,
            "shock_scale": shock_scale.tolist(),
            "topic_noise_scale": s_l.tolist(),
            "attention_mean_level": m.tolist(),
            "clipped_fraction": float(clipped_fraction),
            "n_relevant_effective": int(np.count_nonzero(np.linalg.norm(A_true, axis=1) > 0)),
            "class_counts": {spec.name: int(c) for spec, c in zip(cfg.asset_classes, class_counts)},
            "observed_fraction": float(observed.mean()),
            "factor_structure": bool(np.any(beta_full != 0.0)),
        },
    )
    attention = AttentionData(levels=pd.DataFrame(theta, index=calendar, columns=topic_ids), topic_labels=topic_labels)
    returns = ReturnsData(returns=pd.DataFrame(r, index=calendar, columns=asset_ids), asset_meta=asset_meta)
    return SimulatedData(attention=attention, returns=returns, truth=truth, config=cfg)


# ---------------------------------------------------------------------------
# Step 1 helpers: calendar and factor moments
# ---------------------------------------------------------------------------
def _period_index(calendar: pd.DatetimeIndex, period: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Period id per day (0..T-1, ascending) and the last trading day of each period.

    ``period`` is a pandas offset alias; end-of-period aliases such as ``"ME"``
    are mapped to the equivalent ``Period`` frequency (``"M"``).
    """
    freq = _PERIOD_ALIAS_TO_PERIOD_FREQ.get(period.upper(), period)
    codes, _ = pd.factorize(calendar.to_period(freq), sort=True)
    codes = np.asarray(codes, dtype=np.int64)
    last = np.r_[np.flatnonzero(np.diff(codes) != 0), len(codes) - 1]
    return codes, pd.DatetimeIndex(calendar[last])


def _periods_per_year(period: str) -> float | None:
    """Nominal number of periods per year of an offset alias, or ``None`` if unknown."""
    key = period.upper().split("-")[0]
    return _PERIODS_PER_YEAR.get(key)


def _factor_moments(cfg: SimulationConfig, ann: float) -> tuple[np.ndarray, np.ndarray]:
    """Population period moments of the factors (Part D step 1).

    ``Sigma_ff_period = diag(vol_k^2 / ann)`` with ``vol_k`` the annual factor
    volatilities (``factor_vol_annual`` padded with its last entry / truncated
    to ``K``), and ``mu_f_period = c * Sigma_ff_period^{1/2} 1_K`` with ``c``
    such that ``sqrt(ann * mu' Sigma^-1 mu) = mve_sharpe_annual``. For a
    diagonal ``Sigma`` this gives ``mu_k = sigma_k * SR / sqrt(ann * K)``:
    every factor carries the same per-period Sharpe ratio ``SR / sqrt(ann K)``.
    """
    vols = np.asarray(cfg.factor_vol_annual, dtype=float).ravel()
    if vols.size == 0:
        raise ValueError("factor_vol_annual must have at least one entry")
    if np.any(vols <= 0):
        raise ValueError("factor_vol_annual entries must be > 0")
    vols = _pad(vols, cfg.K, fill=float(vols[-1]))
    Sigma_p = np.diag(vols**2 / ann)
    c = cfg.mve_sharpe_annual / np.sqrt(ann * cfg.K)
    mu_p = c * np.sqrt(np.diag(Sigma_p))
    return mu_p, Sigma_p


def _draw_factors(rng: np.random.Generator, n_days: int, mu_d: np.ndarray, Sigma_d: np.ndarray, phi: float) -> np.ndarray:
    """Daily factors ``f_tau ~ N(mu_d, Sigma_d)``, optionally AR(1) (Part D step 1).

    With ``phi != 0``: ``f_tau - mu_d = phi (f_{tau-1} - mu_d) + u_tau``,
    ``u_tau ~ N(0, (1 - phi^2) Sigma_d)``, started from the stationary law, so
    the unconditional mean and covariance are ``mu_d`` and ``Sigma_d`` for any
    ``|phi| < 1``. ``Sigma_d`` is diagonal, so the draws are independent across
    factors. (Period sums of an AR(1) have a covariance larger than
    ``d Sigma_d`` when ``phi > 0``; that is the point of the option.)
    """
    sd = np.sqrt(np.diag(Sigma_d))
    K = sd.size
    if phi == 0.0:
        return mu_d[None, :] + rng.standard_normal((n_days, K)) * sd[None, :]
    if not abs(phi) < 1.0:
        raise ValueError("factor_ar1 must lie in (-1, 1)")
    dev = _ar1(rng, (n_days, K), phi, np.sqrt(1.0 - phi**2) * sd)
    return mu_d[None, :] + dev


# ---------------------------------------------------------------------------
# Step 3-4 helpers: topics and attention
# ---------------------------------------------------------------------------
def _draw_topic_noise(
    rng: np.random.Generator, n_days: int, sd: np.ndarray, phi: float, iid_mask: np.ndarray
) -> np.ndarray:
    """Topic noise ``eta`` (Part D step 3): i.i.d. normal, or AR(1) with coefficient ``phi``.

    Column ``l`` has stationary standard deviation ``sd[l]`` in both cases.
    Columns flagged in ``iid_mask`` (the placebo topics) are i.i.d. regardless
    of ``phi``; they are overwritten by the variance-matched placebo draws
    afterwards, so this only fixes the consumption of the random stream.
    """
    L = sd.size
    if phi == 0.0:
        return rng.standard_normal((n_days, L)) * sd[None, :]
    if not abs(phi) < 1.0:
        raise ValueError("topic_noise_ar1 must lie in (-1, 1)")
    eta = _ar1(rng, (n_days, L), phi, np.sqrt(1.0 - phi**2) * sd)
    if np.any(iid_mask):
        eta[:, iid_mask] = rng.standard_normal((n_days, int(iid_mask.sum()))) * sd[iid_mask][None, :]
    return eta


def _attention_levels(
    model: str, m: np.ndarray, s_rel: np.ndarray, zeta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Map relative shocks to attention levels (Part D step 4).

    ``additive``: ``theta_{l,tau} = m_l + s_{l,tau} + z_{l,tau}`` with the slow
    component ``s = m_l * s_rel`` and the shock ``z = m_l * zeta`` both
    proportional to the topic's mean level ``m_l``; levels below
    :data:`CLIP_FLOOR` are clipped. ``softmax``: ``theta_tau = softmax(log m + s_rel + zeta)``
    row by row, which puts every day on the simplex.

    Returns ``(theta, shock_scale, clipped_fraction)`` where ``shock_scale``
    converts relative shocks to level units (``m`` for additive, the
    zero-shock level ``m / sum(m)`` for softmax, i.e. the first-order
    derivative ``d theta_l / d zeta_l`` at ``s = zeta = 0`` ignoring the
    common simplex term).
    """
    if model == "additive":
        theta = m[None, :] * (1.0 + s_rel + zeta)
        clipped = theta < CLIP_FLOOR
        theta = np.where(clipped, CLIP_FLOOR, theta)
        return theta, m.copy(), float(clipped.mean())
    if model == "softmax":
        v = np.log(m)[None, :] + s_rel + zeta
        v -= v.max(axis=1, keepdims=True)
        e = np.exp(v)
        theta = e / e.sum(axis=1, keepdims=True)
        return theta, m / m.sum(), 0.0
    raise ValueError(f"unknown attention_model {model!r}")


# ---------------------------------------------------------------------------
# Step 5-6 helpers: assets
# ---------------------------------------------------------------------------
def _class_counts(classes: tuple[AssetClassSpec, ...], n_assets: int) -> np.ndarray:
    """Number of assets per class: shares renormalised, rounded, remainder to the largest class."""
    shares = np.asarray([spec.share for spec in classes], dtype=float)
    if np.any(shares < 0) or shares.sum() <= 0:
        raise ValueError("asset class shares must be >= 0 with a positive sum")
    shares = shares / shares.sum()
    counts = np.round(shares * n_assets).astype(int)
    counts[int(np.argmax(shares))] += n_assets - int(counts.sum())
    if np.any(counts < 0):  # pragma: no cover - only with pathological shares
        raise ValueError("asset class rounding produced a negative count")
    return counts


def _draw_idiosyncratic(rng: np.random.Generator, shape: tuple[int, ...], df: float | None) -> np.ndarray:
    """Unit-variance idiosyncratic innovations: standard normal, or Student-t(df) scaled by ``sqrt((df-2)/df)``."""
    if df is None:
        return rng.standard_normal(shape)
    if df <= 2:
        raise ValueError("fat_tails_df must be > 2 for a finite variance")
    return rng.standard_t(df, size=shape) * np.sqrt((df - 2.0) / df)


def _entry_exit(rng: np.random.Generator, n_days: int, n_assets: int, fraction: float) -> np.ndarray:
    """Boolean ``(n_days, N)`` universe membership (Part D step 6).

    ``round(fraction * N)`` randomly chosen assets are unbalanced: the first
    half enter at a uniformly random day ``in [1, n_days - 1]`` (missing the
    days before), the second half exit at a uniformly random day
    ``in [0, n_days - 2]`` (missing the days after). Everyone else is observed
    on every day.
    """
    alive = np.ones((n_days, n_assets), dtype=bool)
    n_unb = int(round(fraction * n_assets))
    if n_unb == 0:
        return alive
    chosen = rng.choice(n_assets, size=n_unb, replace=False)
    n_enter = n_unb // 2
    late = chosen[:n_enter]
    early = chosen[n_enter:]
    day = np.arange(n_days)[:, None]
    if late.size:
        entry = rng.integers(1, n_days, size=late.size)
        alive[:, late] &= day >= entry[None, :]
    if early.size:
        exit_ = rng.integers(0, n_days - 1, size=early.size)
        alive[:, early] &= day <= exit_[None, :]
    return alive


def _systematic_r2(
    beta_bar: np.ndarray, Sigma_p: np.ndarray, idio_vol: np.ndarray, ann: float, beta_ar1: float, beta_innov_sd: float
) -> float:
    """Population systematic R2 of period returns, averaged over assets (Part D step 7).

    Per asset ``R2_i = q_i / (q_i + vol_i^2 / ann)`` with
    ``q_i = beta_bar_i' Sigma_ff_period beta_bar_i + v_b tr(Sigma_ff_period)``,
    where ``v_b = beta_innov_sd^2 / (1 - beta_ar1^2)`` is the stationary
    variance of the loading deviations ``b_{i,t}`` (zero when ``|beta_ar1| >= 1``
    is not stationary, in which case the deviations start at zero); the second
    term is the extra systematic variance from time-varying loadings and is
    small at the defaults.
    """
    q = np.einsum("ik,kl,il->i", beta_bar, Sigma_p, beta_bar)
    if abs(beta_ar1) < 1.0:
        q = q + beta_innov_sd**2 / (1.0 - beta_ar1**2) * np.trace(Sigma_p)
    r2 = q / (q + idio_vol**2 / ann)
    return float(r2.mean())


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _ar1(rng: np.random.Generator, shape: tuple[int, ...], phi: float, innov_sd: float | np.ndarray) -> np.ndarray:
    """Zero-mean AR(1) along axis 0: ``y_t = phi y_{t-1} + innov_sd * N(0, 1)``.

    Started from the stationary distribution ``N(0, innov_sd^2 / (1 - phi^2))``
    when ``|phi| < 1``, from zero otherwise (random walk or explosive).
    ``innov_sd`` broadcasts against ``shape[1:]``. ``phi = 0`` returns i.i.d. draws.
    """
    innov_sd = np.asarray(innov_sd, dtype=float)
    u = rng.standard_normal(shape) * innov_sd
    if phi == 0.0:
        return u
    y = np.empty(shape)
    if abs(phi) < 1.0:
        y[0] = u[0] / np.sqrt(1.0 - phi**2)
    else:
        y[0] = 0.0
    for t in range(1, shape[0]):
        y[t] = phi * y[t - 1] + u[t]
    return y


def _pad(v: np.ndarray, n: int, fill: float) -> np.ndarray:
    """Pad ``v`` with ``fill`` (or truncate) to length ``n``."""
    v = np.asarray(v, dtype=float).ravel()
    if v.size >= n:
        return v[:n].copy()
    return np.concatenate([v, np.full(n - v.size, fill)])


def _infer_scenario(cfg: SimulationConfig) -> str:
    """Most specific named scenario whose overrides ``cfg`` already satisfies.

    ``no_factor`` is checked before ``topic_null`` (it is the more specific
    of the two: zero signal *and* zero loadings); ``baseline`` always matches.
    """
    for name in ("no_factor", "topic_null", "weak", "softmax", "balanced", "baseline"):
        if scenario_config(name, base=cfg) == cfg:
            return name
    return "custom"  # pragma: no cover - baseline always matches
