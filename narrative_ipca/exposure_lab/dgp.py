"""Data-generating process of the topic-sensitivity lab: topics built from prices (DESIGN.md G.5; D60-D64, D71).

Owner decision D60: asset returns stay as they are (real or artificial) and
each topic's attention is built from the returns of its linked assets plus
noise. Symbols (G.5): ``n = 1..N`` assets, ``k = 1..L`` topics, ``t`` days of
the weekday calendar, ``l`` the lead in days.

Steps of :func:`simulate_lab` / :func:`simulate_from_links`:

1. **Standardised returns.** ``mu_n`` and ``sigma_n`` are asset ``n``'s
   full-sample mean and standard deviation (``ddof = 0``, missing days
   skipped; ``sigma_n = 0`` is replaced by 1); ``rt = (r - mu) / sigma`` the
   standardised return (missing where ``r`` is missing); ``rt_c`` the same
   clipped at plus or minus ``clip_sd`` with missing days set to 0, which is
   what enters the topics.
2. **Moments.** ``V`` (``N x N``) is the population covariance of ``rt_c``
   over all days; ``M`` (``N x N``) the pairwise-complete population
   cross-covariance of ``rt_c`` (rows) with ``rt`` (columns). G.5.1 writes
   ``W_k R W_k'``; ``V`` is used instead because it is the covariance of what
   actually enters the shocks, so ``Var(s_k) = 1`` holds exactly in sample
   (``V = R`` when no day is clipped or missing). ``M`` differs from ``V``
   only through clipping and missing days.
3. **Feasibility (D61).** ``q_k = W_k V W_k'`` is the variance of topic
   ``k``'s signal part; if ``q_k`` exceeds ``feasibility_cap`` the row is
   multiplied by ``sqrt(cap / q_k)``. The news-noise standard deviation is
   ``sigma_u_k = sqrt(1 - W_k V W_k')`` (at least ``sqrt(1 - cap)``).
4. **Designed shocks (G.5.1).** ``s_t = W rt_c_{t+l} + sigma_u * u_t`` with
   ``u`` i.i.d. Student-t(``noise_df``) noise scaled to unit variance
   (Gaussian when ``noise_df = 0``); rows ``t + l`` beyond the sample
   contribute 0 (noise only).
5. **Attention levels (G.5.2).** ``a_{k,t} = m_k + g_{k,t} + kappa * s_{k,t}``
   clipped below at ``1e-6``: base level ``m_k ~ U(level_low, level_high)``,
   slow AR(1) component ``g`` with coefficient ``phi = slow_ar1`` and
   innovation standard deviation ``sigma_eta = slow_sd_ratio * kappa``,
   started from its stationary distribution.
6. **Population truth (G.5.3)** on the observed standardised shocks, via
   :func:`truth_for_window`.

Randomness (D71): the noise ``u`` uses stream 202 and the attention
components ``m`` and ``g`` stream 303, each through the per-topic sub-stream
``default_rng([ExposureConfig.noise_seed, stream, *code points of topic_id])``
(:func:`~narrative_ipca.exposure_lab.links.topic_rng`); the random links use
``ExposureConfig.seed`` (stream 101). A topic's noise and attention
components therefore do not change when exposure values, links, the link
seed or the other topics of the run change.

Validity boundaries
-------------------
* The truth uses the full-sample moments of the filtered noise-free signal
  (G.5.3), so it keeps the serial correlation of daily returns. It is exact
  for a stationary process with these moments; it does not model volatility
  clustering or correlations that change over time (G.12 point 3).
* "Population" moments ``V``, ``M``, ``mu``, ``sigma`` are the full-sample
  moments of the given returns: the realised return sample is the
  population the topics are built from.
* An asset with missing returns gets its truth from the moments over its own
  observed days (assets grouped by observation mask, D78), which is what a
  regression on its observed days estimates.
* The truth ignores the clipping of attention at ``1e-6``; the clipped share
  is recorded in ``meta["clipped_share"]`` (zero or near zero at defaults).
"""

from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import lfilter

from ..config import ShockConfig
from ..shocks import attention_shocks
from .config import AttentionConfig, LabConfig
from .links import LINK_STREAM, build_link_map, build_topic_table, design_matrix, topic_rng
from .types import LinkMap, MarketData, ObservedShocks, SimData, SimTruth, TopicTable

logger = logging.getLogger(__name__)

__all__ = [
    "NOISE_STREAM",
    "ATTENTION_STREAM",
    "ATTENTION_FLOOR",
    "CLIP_WARN_SHARE",
    "simulate_lab",
    "simulate_from_links",
    "truth_for_window",
    "observed_shocks",
    "attenuation",
    "topic_noise",
    "observation_groups",
]

#: Stream ids of the topic noise ``u`` and of the attention components ``m, g`` (D71).
NOISE_STREAM = 202
ATTENTION_STREAM = 303

#: Lower clip of the attention level (G.5.2).
ATTENTION_FLOOR = 1e-6

#: Share of clipped attention cells above which the simulation logs a warning.
CLIP_WARN_SHARE = 1e-3


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def simulate_lab(cfg: LabConfig, market: MarketData | None = None) -> SimData:
    """Simulate topic attention from asset returns (G.5).

    Parameters
    ----------
    cfg:
        The lab configuration; the stage depends on ``universe`` (only when
        ``market`` is ``None``), ``topics``, ``exposure``, ``attention`` and
        ``window.shock_window`` (the window of the stored truth).
    market:
        Asset returns and the asset table; built with
        :func:`~narrative_ipca.exposure_lab.market.build_market` from
        ``cfg.universe`` when ``None``.

    Returns
    -------
    SimData
        Attention levels, designed shocks, the topic table, the link map and
        the population truth for ``cfg.window.shock_window``.
    """
    if market is None:
        from .market import build_market

        market = build_market(cfg.universe)
    t0 = time.perf_counter()
    topics = build_topic_table(cfg.topics)
    links = build_link_map(topics, market.assets, cfg.topics, cfg.exposure)
    t_links = time.perf_counter() - t0
    sim = simulate_from_links(cfg, market, topics, links)
    sim.meta["timings"]["topics_and_links"] = t_links
    return sim


def simulate_from_links(cfg: LabConfig, market: MarketData, topics: TopicTable, links: LinkMap) -> SimData:
    """Simulate attention for a given topic table and link map (G.5 steps 1-6).

    :func:`simulate_lab` builds ``topics`` and ``links`` from ``cfg`` and calls
    this function; call it directly to simulate an edited or hand-built link
    map. Uses ``cfg.exposure`` (tier values, ``lead_days``, ``noise_df``,
    ``noise_seed``), ``cfg.attention`` and ``cfg.window.shock_window``.

    Returns
    -------
    SimData
        ``attention`` and ``designed_shocks`` are ``(n_days, L)`` frames on the
        market calendar with columns ``topic_id``. ``meta`` holds ``V`` and
        ``M`` (asset x asset frames), ``mu`` and ``sigma`` (the
        standardisation used), ``attention_cfg`` (dict), ``base_level``
        (``m_k``), ``feasibility_q`` (``W_unscaled V W_unscaled'`` per topic),
        ``feasibility_scaled_topics``, ``clipped_share``, ``n_clipped``,
        ``n_links``, ``seeds`` and ``timings`` (seconds).
    """
    timings: dict[str, float] = {}
    t_start = time.perf_counter()
    acfg, ecfg = cfg.attention, cfg.exposure
    topic_ids = topics.ids
    asset_ids = [str(a) for a in market.assets.index]
    calendar = market.calendar
    n_days, n_topics = len(calendar), len(topic_ids)
    if n_topics < 1:
        raise ValueError("simulate_from_links: no topics")
    if n_days <= int(cfg.window.shock_window) + 1:
        raise ValueError(f"simulate_from_links: {n_days} days is too short for shock window {cfg.window.shock_window}")

    t = time.perf_counter()
    W_unscaled = design_matrix(links, topics, market.assets, ecfg)
    timings["design"] = time.perf_counter() - t

    # 1-2) standardised returns and moments
    t = time.perf_counter()
    mu, sigma, sigma_raw, rt, obs = _standardise(market.returns)
    rt_c = np.where(obs, np.clip(rt, -float(acfg.clip_sd), float(acfg.clip_sd)), 0.0)
    V = np.atleast_2d(np.cov(rt_c, rowvar=False, ddof=0))
    M = _cross_cov(rt_c, rt, obs)
    R = _pairwise_corr(rt, obs)
    timings["moments"] = time.perf_counter() - t

    # 3) feasibility (D61)
    Wu = W_unscaled.to_numpy(dtype=float)
    q = np.einsum("kn,nm,km->k", Wu, V, Wu)
    cap = float(acfg.feasibility_cap)
    scale = np.ones(n_topics)
    over = q > cap
    scale[over] = np.sqrt(cap / q[over])
    W = Wu * scale[:, None]
    q_after = np.einsum("kn,nm,km->k", W, V, W)
    sigma_u = np.sqrt(np.maximum(1.0 - q_after, 1.0 - cap))
    scaled_topics = [topic_ids[k] for k in np.flatnonzero(over)]
    if scaled_topics:
        logger.warning(
            "simulate: feasibility scaling (W V W' > %.2f) applied to %d topics: %s",
            cap, len(scaled_topics), ", ".join(f"{topic_ids[k]} ({q[k]:.2f})" for k in np.flatnonzero(over)[:10]),
        )

    # 4) designed shocks (G.5.1)
    t = time.perf_counter()
    lead = int(ecfg.lead_days)
    x_lead = np.zeros_like(rt_c)
    x_lead[: n_days - lead] = rt_c[lead:]
    noise_seed = int(ecfg.noise_seed)
    u = topic_noise(topic_ids, n_days, float(ecfg.noise_df), noise_seed)
    signal = x_lead @ W.T  # p_t = W rt_c_{t+l}: the noise-free part of the designed shock
    s = signal + u * sigma_u[None, :]
    timings["shocks"] = time.perf_counter() - t

    # 5) attention levels (G.5.2)
    t = time.perf_counter()
    base, slow = _attention_components(topic_ids, n_days, acfg, noise_seed)
    raw = base[None, :] + slow + float(acfg.kappa) * s
    clipped = raw < ATTENTION_FLOOR
    levels = np.maximum(raw, ATTENTION_FLOOR)
    n_clipped = int(clipped.sum())
    clipped_share = float(n_clipped / clipped.size)
    if n_clipped:
        logger.log(
            logging.WARNING if clipped_share > CLIP_WARN_SHARE else logging.INFO,
            "simulate: %d attention cells (%.4f%%) clipped at %g", n_clipped, 100 * clipped_share, ATTENTION_FLOOR,
        )
    timings["attention"] = time.perf_counter() - t

    # 6) truth (G.5.3)
    t = time.perf_counter()
    t_index = pd.Index(topic_ids, name="topic_id")
    a_index = pd.Index(asset_ids, name="asset_id")
    W_df = pd.DataFrame(W, index=t_index, columns=a_index)
    V_df = pd.DataFrame(V, index=a_index, columns=a_index)
    M_df = pd.DataFrame(M, index=a_index, columns=a_index)
    R_df = pd.DataFrame(R, index=a_index, columns=a_index)
    truth = _population_truth(
        W_unscaled=W_unscaled.copy(),
        W=W_df,
        scale=pd.Series(scale, index=t_index, name="feasibility_scale"),
        sigma_u=pd.Series(sigma_u, index=t_index, name="sigma_u"),
        asset_vol=pd.Series(sigma_raw, index=a_index, name="asset_vol"),
        R=R_df,
        signal=signal,
        rt=rt,
        obs=obs,
        lead=lead,
        kappa=float(acfg.kappa),
        slow_ar1=float(acfg.slow_ar1),
        slow_sd_ratio=float(acfg.slow_sd_ratio),
        shock_window=int(cfg.window.shock_window),
    )
    timings["truth"] = time.perf_counter() - t

    frame = dict(index=pd.DatetimeIndex(calendar), columns=t_index)
    seed = int(ecfg.seed)
    meta = {
        "V": V_df,
        "M": M_df,
        "signal": signal,
        "rt": rt,
        "obs": obs,
        "mu": pd.Series(mu, index=a_index, name="mu"),
        "sigma": pd.Series(sigma, index=a_index, name="sigma"),
        "attention_cfg": dataclasses.asdict(acfg),
        "base_level": pd.Series(base, index=t_index, name="base_level"),
        "feasibility_q": pd.Series(q, index=t_index, name="feasibility_q"),
        "feasibility_scaled_topics": scaled_topics,
        "clipped_share": clipped_share,
        "n_clipped": n_clipped,
        "n_links": int(len(links.table)),
        "lead_days": lead,
        "seeds": {
            "universe": int(cfg.universe.seed),
            "links": [seed, LINK_STREAM],
            "noise": [noise_seed, NOISE_STREAM],
            "attention": [noise_seed, ATTENTION_STREAM],
        },
        "timings": timings,
    }
    timings["total"] = time.perf_counter() - t_start
    logger.info(
        "simulate: %d days x %d assets x %d topics, %d links, lead %d, %d topics scaled, clipped share %.2e (%.2fs)",
        n_days, len(asset_ids), n_topics, meta["n_links"], lead, len(scaled_topics), clipped_share, timings["total"],
    )
    return SimData(
        market=market,
        topics=topics,
        links=links,
        attention=pd.DataFrame(levels, **frame),
        designed_shocks=pd.DataFrame(s, **frame),
        truth=truth,
        lead_days=lead,
        meta=meta,
    )


def truth_for_window(sim: SimData, shock_window: int) -> SimTruth:
    """Population truth on the observed standardised shocks for shock window ``w`` (G.5.3, D62, D63).

    The observed shock is ``z_t = kappa q_t + kappa e_t + D_t`` where

    * ``q_t = p_t - (1/w) sum_{j=1..w} p_{t-j}`` is the trailing-mean filter of
      the noise-free designed signal ``p_t = W rt_c_{t+l}``;
    * ``e_t = sigma_u (u_t - (1/w) sum_j u_{t-j})`` is the filtered news noise,
      with variance ``(1 + 1/w) sigma_u^2``, independent of everything else;
    * ``D_t = g_t - (1/w) sum_j g_{t-j}`` is the filtered slow component, with
      variance ``gamma_0 - (2/w) sum_j gamma_j + (1/w^2) sum_{i,j} gamma_{|i-j|}``
      and ``gamma_h = sigma_eta^2 phi^h / (1 - phi^2)``.

    Hence ``Var(z) = kappa^2 Cov(q) + kappa^2 (1 + 1/w) diag(sigma_u^2) + Var(D) I``
    and ``Cov(z_t, rt_{t+l}) = kappa Cov(q_t, rt_{t+l})``, both from the sample
    moments of the filtered signal. This keeps the serial correlation of
    daily returns (spreads priced at different closes have lag-1
    autocorrelations down to -0.5); the first closed form of DESIGN.md G.5.3,
    which assumed serially uncorrelated returns, missed it. ``S_z`` is
    ``Var(z)`` scaled to a unit diagonal, ``C`` the covariance of the
    standardised shocks with the standardised returns, ``B_true = S_z^{-1} C``
    and ``r2_true_n = C_n' S_z^{-1} C_n``. ``attenuation_k`` is the
    correlation of ``z_k`` with the designed shock ``s_k``.

    Missing returns (D78): for an asset observed on only some paired days,
    ``Cov(q)`` in its ``S_z`` and ``C`` are taken over those days (assets
    with the same observation mask share one solve), while the shocks keep
    their all-day standardisation ``sd(z)``. ``B_true`` is then the
    regression coefficient of the asset's standardised return on the
    standardised shocks over its observed days. With complete data every
    asset uses all days and the stored ``S_z`` is exact for all of them.

    Parameters
    ----------
    sim:
        Output of :func:`simulate_lab`. ``W``, ``W_unscaled``, the feasibility
        scale, ``sigma_u``, ``asset_vol`` and ``R`` come from ``sim.truth``;
        the signal ``p``, the standardised returns and the attention
        parameters from ``sim.meta``.
    shock_window:
        ``w`` of the observed shocks (D9), >= 1.
    """
    tr = sim.truth
    acfg = sim.meta["attention_cfg"]
    return _population_truth(
        W_unscaled=tr.W_unscaled,
        W=tr.W,
        scale=tr.feasibility_scale,
        sigma_u=tr.sigma_u,
        asset_vol=tr.asset_vol,
        R=tr.R,
        signal=sim.meta["signal"],
        rt=sim.meta["rt"],
        obs=sim.meta["obs"],
        lead=int(sim.lead_days),
        kappa=float(acfg["kappa"]),
        slow_ar1=float(acfg["slow_ar1"]),
        slow_sd_ratio=float(acfg["slow_sd_ratio"]),
        shock_window=int(shock_window),
    )


def observed_shocks(
    attention: pd.DataFrame,
    shock_window: int,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
) -> ObservedShocks:
    """Observed shocks ``z`` (D9) and their training-window standardisation ``s_hat`` (G.5.3, D62, D65).

    ``z_{k,t} = a_{k,t} - (1/w) sum_{j=1..w} a_{k,t-j}`` is computed by
    :func:`narrative_ipca.shocks.attention_shocks` (unstandardised); ``scale_k``
    is the population standard deviation (``ddof = 0``, missing values
    skipped) of ``z_k`` over the days ``train_start <= t <= train_end``, with
    zero or undefined values replaced by 1; ``s_hat = z / scale``.

    Parameters
    ----------
    attention:
        ``(n_days, L)`` attention levels (``SimData.attention``).
    shock_window:
        ``w`` >= 1.
    train_start, train_end:
        Training window (inclusive); only it enters ``scale``.

    Returns
    -------
    ObservedShocks
        The first ``w`` rows of ``z`` and ``s_hat`` are ``NaN``.
    """
    w = int(shock_window)
    z = attention_shocks(attention, ShockConfig(window=w, standardize=False)).z
    ts, te = pd.Timestamp(train_start), pd.Timestamp(train_end)
    in_train = (z.index >= ts) & (z.index <= te)
    if not in_train.any():
        logger.warning("observed_shocks: no day in the training window [%s, %s]; scale set to 1", ts.date(), te.date())
    sd = z.loc[in_train].std(axis=0, ddof=0)
    ok = np.isfinite(sd) & (sd > 0.0)
    scale = sd.where(ok, 1.0).astype(float)
    scale.name = "scale"
    n_fixed = int((~ok).sum())
    if n_fixed:
        logger.warning("observed_shocks: %d topics with zero or undefined training std; scale set to 1", n_fixed)
    s_hat = z.div(scale, axis=1)
    return ObservedShocks(z=z, s_hat=s_hat, scale=scale, window=w, train_start=ts, train_end=te)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def attenuation(kappa: float, slow_ar1: float, slow_sd_ratio: float, shock_window: int) -> float:
    """Attenuation ``a = kappa / sqrt(v)`` of the observed shock with window ``w`` (G.5.3).

    ``v = kappa^2 (1 + 1/w) + Var(D_g)`` is the variance of the observed shock
    ``z`` when the designed shock ``s`` has unit variance and no serial
    correlation; ``D_g = g_t - (1/w) sum_{j=1..w} g_{t-j}`` is the slow AR(1)
    component's contribution, with autocovariance ``gamma_h = sigma_eta^2
    phi^h / (1 - phi^2)``, ``phi = slow_ar1`` and ``sigma_eta = slow_sd_ratio
    * kappa``. ``a`` is the correlation of ``z`` with ``s``: about
    ``1 / sqrt(1 + 1/w)`` (0.71 for ``w = 1``, 0.91 for ``w = 5``).
    """
    w = int(shock_window)
    if w < 1:
        raise ValueError("shock_window must be >= 1")
    kappa = float(kappa)
    v = kappa**2 * (1.0 + 1.0 / w) + _var_filtered_slow(kappa, slow_ar1, slow_sd_ratio, w)
    return float(kappa / np.sqrt(v))


def _var_filtered_slow(kappa: float, slow_ar1: float, slow_sd_ratio: float, w: int) -> float:
    """``Var(D_g)`` of the trailing-mean filtered slow AR(1) component (G.5.3), at least 0.

    ``D_g = g_t - (1/w) sum_{j=1..w} g_{t-j}`` with autocovariance
    ``gamma_h = sigma_eta^2 phi^h / (1 - phi^2)``, ``phi = slow_ar1`` and
    ``sigma_eta = slow_sd_ratio * kappa``:
    ``Var(D_g) = gamma_0 - (2/w) sum_{j=1..w} gamma_j + (1/w^2) sum_{i,j=1..w} gamma_{|i-j|}``.
    """
    phi = float(slow_ar1)
    sd_eta = float(slow_sd_ratio) * float(kappa)
    h = np.arange(w + 1, dtype=float)
    gamma = sd_eta**2 * phi**h / (1.0 - phi**2)
    lags = np.arange(1, w)
    sum_ij = w * gamma[0] + 2.0 * float(np.sum((w - lags) * gamma[lags]))
    return max(gamma[0] - (2.0 / w) * float(gamma[1:].sum()) + sum_ij / w**2, 0.0)


def observation_groups(obs: np.ndarray) -> list[tuple[np.ndarray, list[int]]]:
    """Group the columns of the boolean ``obs`` by identical row masks: ``[(row_mask, [column, ...]), ...]``.

    Groups are in the order of their first column. Used for per-asset moments
    over observed days (the truth here, the direct fit in :mod:`.direct`).
    """
    groups: dict[bytes, tuple[np.ndarray, list[int]]] = {}
    for j in range(obs.shape[1]):
        key = np.packbits(obs[:, j]).tobytes()
        if key not in groups:
            groups[key] = (obs[:, j].copy(), [])
        groups[key][1].append(j)
    return list(groups.values())


def topic_noise(topic_ids: list[str], n_days: int, noise_df: float, seed: int) -> np.ndarray:
    """News noise ``u`` (``n_days x L``): unit-variance Student-t, or Gaussian when ``noise_df == 0`` (G.5.1).

    Column ``k`` comes from the sub-stream ``[seed, 202, *code points of
    topic_ids[k]]`` (D71): Student-t draws with ``noise_df`` degrees of
    freedom times ``sqrt((noise_df - 2) / noise_df)``.
    """
    df = float(noise_df)
    out = np.empty((int(n_days), len(topic_ids)), dtype=float)
    for k, tid in enumerate(topic_ids):
        rng = topic_rng(seed, NOISE_STREAM, tid)
        if df == 0.0:
            out[:, k] = rng.standard_normal(int(n_days))
        else:
            out[:, k] = rng.standard_t(df, size=int(n_days)) * np.sqrt((df - 2.0) / df)
    return out


def _attention_components(
    topic_ids: list[str], n_days: int, acfg: AttentionConfig, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Base levels ``m`` (``L``) and slow AR(1) components ``g`` (``n_days x L``) of G.5.2.

    Per topic, from the sub-stream ``[seed, 303, *code points of topic_id]``:
    ``m ~ U(level_low, level_high)``, then ``n_days`` standard normals ``e``;
    ``g_0 = e_0 * sigma_eta / sqrt(1 - phi^2)`` (stationary start) and
    ``g_t = phi g_{t-1} + sigma_eta e_t``.
    """
    n_days = int(n_days)
    phi = float(acfg.slow_ar1)
    sd_eta = float(acfg.slow_sd_ratio) * float(acfg.kappa)
    base = np.empty(len(topic_ids), dtype=float)
    innov = np.empty((n_days, len(topic_ids)), dtype=float)
    for k, tid in enumerate(topic_ids):
        rng = topic_rng(seed, ATTENTION_STREAM, tid)
        base[k] = rng.uniform(float(acfg.level_low), float(acfg.level_high))
        innov[:, k] = rng.standard_normal(n_days)
    innov *= sd_eta
    innov[0] /= np.sqrt(1.0 - phi**2)
    slow = lfilter([1.0], [1.0, -phi], innov, axis=0) if n_days else innov
    return base, np.asarray(slow, dtype=float)


def _standardise(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full-sample standardisation of G.5: ``(mu, sigma, sigma_raw, rt, observed)``.

    ``mu`` and ``sigma_raw`` are the mean and population standard deviation
    over observed (finite) days; ``sigma`` equals ``sigma_raw`` with zero or
    undefined values replaced by 1 (and ``mu`` undefined replaced by 0);
    ``rt = (r - mu) / sigma`` is ``NaN`` where ``r`` is not observed.
    """
    X = returns.to_numpy(dtype=float, copy=True)
    obs = np.isfinite(X)
    X0 = np.where(obs, X, 0.0)
    n = obs.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_raw = X0.sum(axis=0) / n
        mu = np.where(n > 0, mu_raw, 0.0)
        dev = np.where(obs, X - mu, 0.0)
        sigma_raw = np.where(n > 0, np.sqrt((dev**2).sum(axis=0) / n), np.nan)
    sigma = np.where(np.isfinite(sigma_raw) & (sigma_raw > 0.0), sigma_raw, 1.0)
    bad = ~(np.isfinite(sigma_raw) & (sigma_raw > 0.0))
    if bad.any():
        logger.warning(
            "simulate: %d assets with zero or undefined volatility; standardised with sigma = 1", int(bad.sum())
        )
    rt = np.where(obs, (X - mu) / sigma, np.nan)
    return mu, sigma, sigma_raw, rt, obs


def _solve_spd(S: np.ndarray, C: np.ndarray) -> np.ndarray:
    """``S^{-1} C`` for a symmetric positive definite ``S`` (Cholesky; least squares as a fallback).

    ``S`` is positive definite here because the news-noise variance
    ``sigma_u^2 >= 1 - cap > 0`` sits on its diagonal. Cholesky is also far
    faster than LU with the bundled multi-threaded OpenBLAS (0.5 s against
    5 ms at 500 topics).
    """
    try:
        return cho_solve(cho_factor(S, lower=True), C)
    except (np.linalg.LinAlgError, ValueError):
        logger.warning("truth: S_z is not positive definite; using least squares")
        return np.linalg.lstsq(S, C, rcond=None)[0]


def _cross_cov(x: np.ndarray, y: np.ndarray, obs_y: np.ndarray) -> np.ndarray:
    """Pairwise-complete population cross-covariance ``M[i, j] = Cov(x_i, y_j)``.

    ``x`` (``T x N``) is fully observed; for each column ``j`` of ``y`` the
    moments use the days where ``y_j`` is observed. Columns with no observed
    day are 0.
    """
    O = obs_y.astype(float)
    Y0 = np.where(obs_y, y, 0.0)
    n = O.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sxy = x.T @ Y0
        sx = x.T @ O
        sy = Y0.sum(axis=0)
        M = sxy / n[None, :] - (sx / n[None, :]) * (sy / n)[None, :]
    M[:, n == 0] = 0.0
    return M


def _pairwise_corr(z: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Pairwise-complete Pearson correlation of the columns of ``z`` (``NaN`` where fewer than 2 common days)."""
    O = obs.astype(float)
    Z0 = np.where(obs, z, 0.0)
    n = O.T @ O
    sx = Z0.T @ O  # sum of z_i over the days where both i and j are observed
    sxx = (Z0**2).T @ O
    sxy = Z0.T @ Z0
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = sx / n, sx.T / n
        cov = sxy / n - mx * my
        vx = sxx / n - mx**2
        vy = sxx.T / n - my**2
        corr = cov / np.sqrt(vx * vy)
    corr[(n < 2) | ~(vx > 1e-14) | ~(vy > 1e-14)] = np.nan
    corr = np.clip(corr, -1.0, 1.0)
    diag = np.diag(corr).copy()
    np.fill_diagonal(corr, np.where(np.isfinite(diag), 1.0, np.nan))
    return corr


def _population_truth(
    *,
    W_unscaled: pd.DataFrame,
    W: pd.DataFrame,
    scale: pd.Series,
    sigma_u: pd.Series,
    asset_vol: pd.Series,
    R: pd.DataFrame,
    signal: np.ndarray,
    rt: np.ndarray,
    obs: np.ndarray,
    lead: int,
    kappa: float,
    slow_ar1: float,
    slow_sd_ratio: float,
    shock_window: int,
) -> SimTruth:
    """Assemble :class:`SimTruth` from the design and the filtered signal (formulas in :func:`truth_for_window`)."""
    w = int(shock_window)
    if w < 1:
        raise ValueError("shock_window must be >= 1")
    t_index, a_index = W.index, W.columns
    kappa = float(kappa)
    su2 = sigma_u.to_numpy(dtype=float) ** 2
    p = np.asarray(signal, dtype=float)
    n_days, n_topics = p.shape
    lead = int(lead)
    if n_days <= w + lead + 2:
        raise ValueError(f"truth: {n_days} days is too short for shock window {w}")

    # q_t = p_t - mean(p_{t-1..t-w}) for t >= w (trailing, strictly prior rows; D9)
    csum = np.vstack([np.zeros((1, n_topics)), np.cumsum(p, axis=0)])
    trailing = (csum[w:-1] - csum[: -w - 1]) / w
    q = p[w:] - trailing  # rows t = w .. n_days-1

    # pair the shock day t with the return day t + l
    rows = np.arange(w, n_days - lead)
    q_pair = q[: len(rows)]
    y_pair = np.asarray(rt, dtype=float)[rows + lead]
    o_pair = np.asarray(obs, dtype=bool)[rows + lead]

    cov_q = np.atleast_2d(np.cov(q_pair, rowvar=False, ddof=0))
    cov_qy = _cross_cov(q_pair, y_pair, o_pair)  # per asset over its observed paired days
    p_now = p[rows]
    cov_qp = ((q_pair - q_pair.mean(0)) * (p_now - p_now.mean(0))).mean(0)

    # variance of the news noise and of the filtered slow component D_t on the diagonal
    noise_diag = kappa**2 * (1.0 + 1.0 / w) * su2 + _var_filtered_slow(kappa, slow_ar1, slow_sd_ratio, w)
    var_z = kappa**2 * cov_q
    var_z[np.diag_indices(n_topics)] += noise_diag
    sd = np.sqrt(np.diag(var_z))
    S_z = var_z / np.outer(sd, sd)
    np.fill_diagonal(S_z, 1.0)
    C = kappa * cov_qy / sd[:, None]
    a = kappa * (cov_qp + su2) / sd

    # B_true per group of assets with the same observed paired days (D78): Cov(q) over those days,
    # shocks standardised by the all-day sd. With complete data there is one group and S_z itself.
    n_assets = C.shape[1]
    B = np.zeros((n_topics, n_assets))
    for rows_g, cols in observation_groups(o_pair):
        n_g = int(rows_g.sum())
        if n_g < 2:
            continue  # no observed day: C is 0, so B stays 0
        if n_g == len(rows_g):
            S_g = S_z
        else:
            var_g = kappa**2 * np.atleast_2d(np.cov(q_pair[rows_g], rowvar=False, ddof=0))
            var_g[np.diag_indices(n_topics)] += noise_diag
            S_g = var_g / np.outer(sd, sd)
        B[:, cols] = _solve_spd(S_g, C[:, cols])
    r2 = (C * B).sum(axis=0)
    return SimTruth(
        W=W,
        W_unscaled=W_unscaled,
        feasibility_scale=scale,
        sigma_u=sigma_u,
        attenuation=pd.Series(a, index=t_index, name="attenuation"),
        B_true=pd.DataFrame(B, index=t_index, columns=a_index),
        r2_true=pd.Series(r2, index=a_index, name="r2_true"),
        S_z=pd.DataFrame(S_z, index=t_index, columns=t_index),
        asset_vol=asset_vol,
        R=R,
        shock_window=w,
    )
