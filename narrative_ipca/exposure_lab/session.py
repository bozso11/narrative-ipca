"""Cached orchestration of the topic-exposure lab stages (DESIGN.md G.10, G.13; D71).

:class:`LabSession` runs the lab stages in order and memoises each result
under the cache key of the sub-configuration the stage depends on
(:meth:`LabConfig.key`, D71). Changing only the forecast window therefore
re-runs only the evaluation; changing the direct estimator re-runs only the
fit and the evaluation; the BKS panel is reused across BKS estimation
settings. The dashboard (``dashboard/app.py``) holds one session for the
whole server process; :func:`run_lab` runs every stage once for scripts and
tests.

Stages and their keys (``k(stage)`` is ``LabConfig.key(stage)``):

====================  ===========================================  =================================
stage                 key                                          computed by
====================  ===========================================  =================================
``market``            ``k("market")``                              :func:`.market.build_market`
``simulation``        ``k("simulation")``                          :func:`.dgp.simulate_lab`
``truth``             ``k("simulation")`` + shock window ``w``     :func:`.dgp.truth_for_window`
``shocks``            ``k("shocks")``                              :func:`.dgp.observed_shocks`
``direct``            ``k("direct")``                              :func:`.direct.fit_direct`
``evaluation``        ``k("evaluation")``                          :func:`.evaluate.evaluate_window`
``sweep``             ``k("evaluation")``                          :func:`.evaluate.window_sweep`
``bks_panel``         ``k("bks_panel")``                           :func:`.bks.build_bks_panel`
``bks_fit``           ``k("bks_fit")``                             :func:`.bks.fit_bks`
``bks``               ``k("bks_evaluation")``                      :func:`.bks.evaluate_bks`
====================  ===========================================  =================================

Validity boundaries
-------------------
* Results are shared, not copied: callers must not modify what a stage
  returns (the next caller with the same key gets the same object).
* ``SimData.truth`` holds the truth for the shock window of the config that
  first built the simulation (the simulation key does not contain ``w``);
  use :meth:`LabSession.truth` for the truth of a given config.
* Errors propagate and are not cached. Two threads asking for the same
  missing key compute it once: the second waits on a per-key lock and then
  reads the cache. The cache lock is held only for cache reads and writes,
  so a long BKS fit does not block the fast stages of another caller.
* Each stage keeps at most ``max_entries`` results, the BKS stages at most
  ``bks_max_entries`` (a 500 x 500 BKS panel holds hundreds of megabytes);
  least recently used first out.
* ``last_timings`` is per thread: Streamlit runs each browser session's
  script in its own thread, so one session's timings table never shows
  another session's stages.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from .config import LabConfig
from .types import BKSLabResult, DirectFit, MarketData, ObservedShocks, SimData, SimTruth, WindowEval

logger = logging.getLogger(__name__)

__all__ = [
    "SESSION_STAGES",
    "BKS_STAGES",
    "BKS_MAX_ENTRIES",
    "RUN_LAB_KEYS",
    "LabSession",
    "run_lab",
]

#: Stages a :class:`LabSession` memoises, in pipeline order.
SESSION_STAGES: tuple[str, ...] = (
    "market", "simulation", "truth", "shocks", "direct", "evaluation", "sweep", "bks_panel", "bks_fit", "bks",
)

#: Stages whose results can be large (the weekly BKS panel and fit); they get a smaller cache.
BKS_STAGES: frozenset[str] = frozenset({"bks_panel", "bks_fit", "bks"})

#: Default number of cached results per BKS stage.
BKS_MAX_ENTRIES = 2

#: Keys of the dict returned by :func:`run_lab` (the BKS keys only with ``with_bks=True``).
RUN_LAB_KEYS: tuple[str, ...] = (
    "config", "keys", "market", "simulation", "truth", "shocks", "direct", "evaluation", "sweep", "timings",
)

ProgressFn = Callable[[int, int, str], None]


class LabSession:
    """Stage results of the lab memoised by configuration key (G.10, D71).

    Parameters
    ----------
    max_entries:
        Results kept per stage (least recently used evicted first). Six
        covers switching back and forth between a few settings.
    bks_max_entries:
        Results kept per BKS stage (:data:`BKS_STAGES`), at most
        ``max_entries``.

    Attributes
    ----------
    last_timings:
        Per stage, the seconds of the calling thread's last call and whether
        it was served from the cache: ``{stage: {"seconds": float, "cached":
        bool, "key": str}}``.
    stats:
        Per stage counts of cache ``hits`` and ``misses``.
    """

    def __init__(self, max_entries: int = 6, bks_max_entries: int = BKS_MAX_ENTRIES) -> None:
        if int(max_entries) < 1 or int(bks_max_entries) < 1:
            raise ValueError("max_entries and bks_max_entries must be >= 1")
        self.max_entries = int(max_entries)
        self._limits = {
            s: min(int(bks_max_entries), self.max_entries) if s in BKS_STAGES else self.max_entries
            for s in SESSION_STAGES
        }
        self._caches: dict[str, OrderedDict[str, Any]] = {s: OrderedDict() for s in SESSION_STAGES}
        self._lock = threading.RLock()
        self._key_locks: dict[str, threading.RLock] = {}
        self._local = threading.local()
        self.stats: dict[str, dict[str, int]] = {s: {"hits": 0, "misses": 0} for s in SESSION_STAGES}

    @property
    def last_timings(self) -> dict[str, dict[str, Any]]:
        """Timings of the calling thread's last stage calls (see the class docstring)."""
        timings = getattr(self._local, "timings", None)
        if timings is None:
            timings = {}
            self._local.timings = timings
        return timings

    # ------------------------------------------------------------------
    # Cache plumbing
    # ------------------------------------------------------------------
    @staticmethod
    def stage_key(stage: str, cfg: LabConfig) -> str:
        """Cache key of ``stage`` for ``cfg`` (see the module table)."""
        if stage == "truth":
            return f"truth-{cfg.key('simulation')}-w{int(cfg.window.shock_window)}"
        if stage == "sweep":
            return f"sweep-{cfg.key('evaluation')}"
        if stage == "bks":
            return cfg.key("bks_evaluation")
        if stage not in SESSION_STAGES:
            raise KeyError(f"unknown stage {stage!r}; known: {list(SESSION_STAGES)}")
        return cfg.key(stage)

    def has(self, stage: str, cfg: LabConfig) -> bool:
        """``True`` when ``stage`` for ``cfg`` is cached (nothing is computed)."""
        key = self.stage_key(stage, cfg)
        with self._lock:
            return key in self._caches[stage]

    def peek(self, stage: str, cfg: LabConfig) -> Any | None:
        """The cached result of ``stage`` for ``cfg``, or ``None`` (nothing is computed)."""
        key = self.stage_key(stage, cfg)
        with self._lock:
            return self._caches[stage].get(key)

    def clear(self, stage: str | None = None) -> None:
        """Drop the cached results of one stage, or of all stages."""
        with self._lock:
            for s in [stage] if stage is not None else SESSION_STAGES:
                self._caches[s].clear()

    def _hit(self, stage: str, key: str) -> tuple[bool, Any]:
        """Cache lookup under the cache lock; records the hit."""
        with self._lock:
            cache = self._caches[stage]
            if key in cache:
                cache.move_to_end(key)
                self.stats[stage]["hits"] += 1
                self.last_timings[stage] = {"seconds": 0.0, "cached": True, "key": key}
                return True, cache[key]
        return False, None

    def _get(self, stage: str, cfg: LabConfig, compute: Callable[[], Any]) -> Any:
        key = self.stage_key(stage, cfg)
        found, value = self._hit(stage, key)
        if found:
            return value
        lock_id = f"{stage}:{key}"
        with self._lock:
            key_lock = self._key_locks.setdefault(lock_id, threading.RLock())
        # One computation per key: a second caller waits here and then finds the result in the cache.
        # Locks are taken downstream-to-upstream only (a stage never asks for a later stage), so
        # waiting cannot deadlock.
        with key_lock:
            found, value = self._hit(stage, key)
            if found:
                return value
            try:
                t0 = time.perf_counter()
                value = compute()
                seconds = time.perf_counter() - t0
                with self._lock:
                    cache = self._caches[stage]
                    cache[key] = value
                    cache.move_to_end(key)
                    while len(cache) > self._limits[stage]:
                        cache.popitem(last=False)
                    self.stats[stage]["misses"] += 1
                    self.last_timings[stage] = {"seconds": seconds, "cached": False, "key": key}
            finally:
                with self._lock:
                    self._key_locks.pop(lock_id, None)
        logger.info("LabSession: %s computed in %.3fs (%s)", stage, seconds, key)
        return value

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------
    def market(self, cfg: LabConfig) -> MarketData:
        """Asset table and returns of ``cfg.universe`` (G.2)."""
        from .market import build_market

        return self._get("market", cfg, lambda: build_market(cfg.universe))

    def simulation(self, cfg: LabConfig) -> SimData:
        """Topics, links and simulated attention (G.3-G.5), on the cached market."""
        from .dgp import simulate_lab

        return self._get("simulation", cfg, lambda: simulate_lab(cfg, market=self.market(cfg)))

    def truth(self, cfg: LabConfig) -> SimTruth:
        """Population truth for the shock window ``cfg.window.shock_window`` (G.5.3)."""
        from .dgp import truth_for_window

        return self._get("truth", cfg, lambda: truth_for_window(self.simulation(cfg), int(cfg.window.shock_window)))

    def shocks(self, cfg: LabConfig) -> ObservedShocks:
        """Observed shocks and their training-window standardisation (G.5.3, D62)."""
        from .dgp import observed_shocks

        def compute() -> ObservedShocks:
            w = cfg.window
            return observed_shocks(self.simulation(cfg).attention, int(w.shock_window), w.train_start, w.train_end)

        return self._get("shocks", cfg, compute)

    def direct(self, cfg: LabConfig) -> DirectFit:
        """Direct exposure regression on the training window (G.7.1)."""
        from .direct import fit_direct

        return self._get(
            "direct", cfg, lambda: fit_direct(self.simulation(cfg), self.shocks(cfg), cfg.direct, self.truth(cfg))
        )

    def evaluation(self, cfg: LabConfig) -> WindowEval:
        """Out-of-sample evaluation in the forecast window (G.8 points 1-5)."""
        from .evaluate import evaluate_window

        def compute() -> WindowEval:
            sim, shocks, fit, truth = self.simulation(cfg), self.shocks(cfg), self.direct(cfg), self.truth(cfg)
            return evaluate_window(sim, shocks, fit, cfg.window, truth)

        return self._get("evaluation", cfg, compute)

    def sweep(self, cfg: LabConfig) -> Any:
        """OOS R2 over consecutive forecast windows (G.8 point 6); a ``DataFrame``."""
        from .evaluate import window_sweep

        return self._get(
            "sweep",
            cfg,
            lambda: window_sweep(self.simulation(cfg), self.shocks(cfg), self.direct(cfg), cfg.window, self.truth(cfg)),
        )

    def bks_panel(self, cfg: LabConfig) -> Any:
        """Weekly BKS panel (G.7.2); a :class:`~narrative_ipca.exposure_lab.bks.BKSPanel`."""
        from .bks import build_bks_panel

        return self._get(
            "bks_panel", cfg, lambda: build_bks_panel(self.simulation(cfg), cfg.bks, int(cfg.window.shock_window))
        )

    def bks_fit(self, cfg: LabConfig, progress: ProgressFn | None = None) -> Any:
        """Sparse IPCA on the training weeks (G.7.2); a :class:`~narrative_ipca.exposure_lab.bks.BKSFit`.

        ``progress(done, total, message)`` is called during the lambda path
        (only when the fit is computed, not on a cache hit).
        """
        from .bks import fit_bks

        def compute() -> Any:
            w = cfg.window
            return fit_bks(self.bks_panel(cfg), cfg.bks, w.train_end, progress=progress, train_start=w.train_start)

        return self._get("bks_fit", cfg, compute)

    def bks(self, cfg: LabConfig, progress: ProgressFn | None = None) -> BKSLabResult:
        """BKS evaluated in the forecast window, with the per-topic split (G.7.2, G.8; D52)."""
        from .bks import evaluate_bks

        return self._get(
            "bks", cfg, lambda: evaluate_bks(self.bks_panel(cfg), self.bks_fit(cfg, progress=progress), cfg.window)
        )

    def run(self, cfg: LabConfig, with_bks: bool = False, progress: ProgressFn | None = None) -> dict[str, Any]:
        """Run (or fetch) every stage for ``cfg``; see :func:`run_lab` for the returned keys."""
        t0 = time.perf_counter()
        out: dict[str, Any] = {"config": cfg}
        timings: dict[str, dict[str, Any]] = {}
        stages: list[tuple[str, Callable[[], Any]]] = [
            ("market", lambda: self.market(cfg)),
            ("simulation", lambda: self.simulation(cfg)),
            ("truth", lambda: self.truth(cfg)),
            ("shocks", lambda: self.shocks(cfg)),
            ("direct", lambda: self.direct(cfg)),
            ("evaluation", lambda: self.evaluation(cfg)),
            ("sweep", lambda: self.sweep(cfg)),
        ]
        if with_bks:
            stages += [
                ("bks_panel", lambda: self.bks_panel(cfg)),
                ("bks_fit", lambda: self.bks_fit(cfg, progress=progress)),
                ("bks", lambda: self.bks(cfg)),
            ]
        for name, call in stages:
            out[name] = call()
            timings[name] = dict(self.last_timings.get(name, {}))
        out["keys"] = {name: self.stage_key(name, cfg) for name, _ in stages}
        timings["total"] = {"seconds": time.perf_counter() - t0, "cached": False}
        out["timings"] = timings
        return out


def run_lab(cfg: LabConfig, with_bks: bool = False, progress: ProgressFn | None = None) -> dict[str, Any]:
    """Run every lab stage once for ``cfg`` (fresh :class:`LabSession`; G.13).

    Parameters
    ----------
    cfg:
        The lab configuration.
    with_bks:
        Also build the weekly BKS panel, fit Sparse IPCA and evaluate it
        (seconds at 20 topics, minutes at 500; G.10).
    progress:
        Optional ``progress(done, total, message)`` callback for the BKS
        lambda path.

    Returns
    -------
    dict
        ``config`` (the :class:`LabConfig`), ``market``
        (:class:`MarketData`), ``simulation`` (:class:`SimData`), ``truth``
        (:class:`SimTruth` for ``cfg.window.shock_window``), ``shocks``
        (:class:`ObservedShocks`), ``direct`` (:class:`DirectFit`),
        ``evaluation`` (:class:`WindowEval`), ``sweep`` (``DataFrame``),
        ``keys`` (stage -> cache key) and ``timings`` (stage -> ``{"seconds",
        "cached"}``, plus ``total``); with ``with_bks`` also ``bks_panel``,
        ``bks_fit`` and ``bks`` (:class:`BKSLabResult`).
    """
    return LabSession(max_entries=1, bks_max_entries=1).run(cfg, with_bks=with_bks, progress=progress)
