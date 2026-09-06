"""Benchmark the Sparse IPCA solver on the full-size simulated panel.

Usage (from the repository root, no installation needed)::

    python scripts/benchmark_solver.py                 # numba kernel and numpy fallback
    python scripts/benchmark_solver.py --no-fallback   # numba kernel only
    python scripts/benchmark_solver.py --backends numpy --skip-dense
    python scripts/benchmark_solver.py --scenario baseline --seed 1 --n-lambdas 12 --ratio 1e-2

The panel is the DESIGN.md Part D baseline scenario (500 assets, 120 topics,
K = 3, 20 years of monthly periods, about 92,000 asset-period rows). For each
requested backend of :mod:`narrative_ipca.grouplasso` the script times

1. ``lambda_max`` (warm-up sweeps plus the KKT threshold, D22),
2. one ``fit_sparse_ipca`` at ``1e-3 * lam_max`` (the dense end: most
   narratives active, the slowest single fit),
3. one fit at ``1e-2 * lam_max``,
4. a full ``lambda_path`` with ``--n-lambdas`` points and ``--ratio``
   (the tuning grid of the OOS loop),

and prints one table per stage plus the agreement between the backends
(relative difference of ``Gamma``, of the objective, and equality of the
selected set), which must be at rounding level: both paths implement the
same algorithm. Wall-clock times are ``time.perf_counter`` seconds on the
current machine; the first numba call of a fresh cache compiles the kernel
(reported separately as "kernel warm-up") and is excluded from the stage
timings.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from narrative_ipca import covariances, data, grouplasso, panel, shocks, sparse_ipca  # noqa: E402
from narrative_ipca.config import CovarianceConfig, DataConfig, EstimationConfig, LambdaGridConfig, ShockConfig  # noqa: E402
from narrative_ipca.simulation import scenario_config, simulate  # noqa: E402
from narrative_ipca.types import IPCAPanel  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_solver",
        description="Time lambda_max, single Sparse IPCA fits and a regularisation path on the full-size simulated panel.",
    )
    parser.add_argument("--scenario", default="baseline", help="simulation scenario (default baseline)")
    parser.add_argument("--seed", type=int, default=None, help="simulation seed (default: the scenario's)")
    parser.add_argument("--n-lambdas", type=int, default=12, help="path length (default 12)")
    parser.add_argument("--ratio", type=float, default=1e-2, help="lam_min / lam_max of the path (default 1e-2)")
    parser.add_argument("--dense-frac", type=float, default=1e-3, help="lambda of the dense single fit as a fraction of lam_max (default 1e-3)")
    parser.add_argument("--mid-frac", type=float, default=1e-2, help="lambda of the second single fit as a fraction of lam_max (default 1e-2)")
    parser.add_argument("--backends", default=None, help="comma-separated subset of numba,numpy (default: both when numba is available)")
    parser.add_argument("--no-fallback", action="store_true", help="skip the numpy reference path")
    parser.add_argument("--skip-dense", action="store_true", help="skip the dense single fit (the slow one on the numpy path)")
    parser.add_argument("--repeat", type=int, default=1, help="repetitions per stage; the minimum time is reported (default 1)")
    parser.add_argument("-v", "--verbose", action="store_true", help="INFO logging from narrative_ipca (default WARNING)")
    return parser


def build_full_panel(scenario: str, seed: int | None) -> tuple[IPCAPanel, float]:
    """The DESIGN.md Part D panel of ``scenario`` and the seconds it took to build."""
    t0 = time.perf_counter()
    sim_cfg = scenario_config(scenario)
    if seed is not None:
        sim_cfg = replace(sim_cfg, seed=int(seed))
    sim = simulate(sim_cfg)
    aligned = data.align_inputs(sim.attention, sim.returns, DataConfig())
    sh = shocks.attention_shocks(aligned.attention, ShockConfig())
    cov = covariances.build_covariance_panel(sh, aligned.returns, CovarianceConfig(), "M")
    pnl = panel.build_panel(cov, aligned.returns, DataConfig(), CovarianceConfig())
    pnl.moments()  # cache the per-period moments so that no stage pays for them
    return pnl, time.perf_counter() - t0


def _timed(fn, repeat: int) -> tuple[float, Any]:
    best = np.inf
    out = None
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def run_backend(name: str, pnl: IPCAPanel, cfg: EstimationConfig, args: argparse.Namespace) -> dict[str, Any]:
    """All stages on one backend; returns timings and results for the agreement check."""
    grouplasso.USE_NUMBA = name == "numba"
    assert grouplasso.active_backend() == name, (name, grouplasso.active_backend())
    rows: dict[str, Any] = {"backend": name}

    if name == "numba":
        # first call compiles or loads the cached kernel; keep it out of the stage timings
        G = np.eye(6)
        b = np.ones(6)
        t0 = time.perf_counter()
        grouplasso.group_lasso_gram(G, b, 3, np.zeros(2), max_iter=5)
        rows["warmup_s"] = time.perf_counter() - t0

    t, lam_max = _timed(lambda: sparse_ipca.lambda_max(pnl, cfg), args.repeat)
    rows["lambda_max"] = {"time_s": t, "value": float(lam_max)}

    fits: dict[str, Any] = {}
    single = [] if args.skip_dense else [("dense", args.dense_frac)]
    single.append(("mid", args.mid_frac))
    for label, frac in single:
        lam = frac * lam_max
        t, res = _timed(lambda: sparse_ipca.fit_sparse_ipca(pnl, cfg, lam=lam), args.repeat)
        fits[label] = res
        rows[f"fit_{label}"] = {
            "time_s": t,
            "frac": frac,
            "sweeps": res.n_iter,
            "inner": int(res.meta["inner_iters"]),
            "selected": res.n_selected,
            "converged": res.converged,
            "objective": res.objective,
        }

    path_cfg = replace(cfg, lam_grid=LambdaGridConfig(n_lambdas=int(args.n_lambdas), ratio=float(args.ratio)))
    t, (points, path_fits) = _timed(lambda: sparse_ipca.lambda_path(pnl, path_cfg), args.repeat)
    rows["path"] = {
        "time_s": t,
        "n_points": len(points),
        "sweeps": int(sum(p.n_iter for p in points)),
        "inner": int(sum(f.meta["inner_iters"] for f in path_fits)),
        "selected": [p.n_selected for p in points],
        "converged": all(p.converged for p in points),
        "objectives": [p.objective for p in points],
    }
    rows["_fits"] = fits
    rows["_path_fits"] = path_fits
    return rows


def _fmt_time(x: float) -> str:
    return f"{x:8.2f} s"


def print_tables(results: list[dict[str, Any]], pnl: IPCAPanel, build_s: float, args: argparse.Namespace) -> None:
    print(f"panel: T={pnl.T} periods, N={len(pnl.assets)} assets, p={pnl.p} instruments (L={pnl.L}), n_obs={pnl.n_obs}, K={EstimationConfig().K}; built in {build_s:.1f} s")
    print(f"numba available: {grouplasso.HAVE_NUMBA} (NARRATIVE_IPCA_NO_NUMBA set: {grouplasso.NUMBA_DISABLED_BY_ENV})")
    for r in results:
        if "warmup_s" in r:
            print(f"kernel warm-up ({r['backend']}, compile or cache load): {r['warmup_s']:.2f} s")
    print()
    names = [r["backend"] for r in results]
    header = f"{'stage':<34}" + "".join(f"{n:>14}" for n in names) + "   detail"
    print(header)
    print("-" * len(header))
    lm = [r["lambda_max"] for r in results]
    print(f"{'lambda_max':<34}" + "".join(f"{_fmt_time(v['time_s']):>14}" for v in lm) + f"   value {lm[0]['value']:.6g}")
    for label, title in (("dense", f"fit at {args.dense_frac:g} * lam_max"), ("mid", f"fit at {args.mid_frac:g} * lam_max")):
        key = f"fit_{label}"
        if key not in results[0]:
            continue
        vals = [r[key] for r in results]
        v0 = vals[0]
        print(f"{title:<34}" + "".join(f"{_fmt_time(v['time_s']):>14}" for v in vals) + f"   {v0['sweeps']} sweeps, {v0['inner']} inner sweeps, {v0['selected']} selected, converged={v0['converged']}")
    vals = [r["path"] for r in results]
    v0 = vals[0]
    path_title = f"path {v0['n_points']} pts, ratio {args.ratio:g}"
    print(f"{path_title:<34}" + "".join(f"{_fmt_time(v['time_s']):>14}" for v in vals) + f"   {v0['sweeps']} sweeps, {v0['inner']} inner sweeps, selected {v0['selected']}, converged={v0['converged']}")
    if len(results) > 1:
        print()
        print("agreement between backends (relative to the first column):")
        ref = results[0]
        for other in results[1:]:
            for label in ("dense", "mid"):
                if label not in ref["_fits"]:
                    continue
                a, b = ref["_fits"][label], other["_fits"][label]
                dg = float(np.max(np.abs(a.Gamma - b.Gamma)) / max(np.max(np.abs(a.Gamma)), 1e-300))
                dobj = abs(a.objective - b.objective) / max(abs(a.objective), 1e-300)
                same = np.array_equal(a.selected, b.selected)
                print(f"  fit {label:<6} {ref['backend']} vs {other['backend']}: max|dGamma|/max|Gamma| = {dg:.2e}, |dobj|/obj = {dobj:.2e}, same selected = {same}, sweeps {a.n_iter}/{b.n_iter}")
            oa = np.asarray(ref["path"]["objectives"])
            ob = np.asarray(other["path"]["objectives"])
            dg_path = max(
                float(np.max(np.abs(fa.Gamma - fb.Gamma)) / max(np.max(np.abs(fa.Gamma)), 1e-300))
                for fa, fb in zip(ref["_path_fits"], other["_path_fits"])
            )
            print(f"  path       {ref['backend']} vs {other['backend']}: max rel objective diff = {float(np.max(np.abs(oa - ob) / np.abs(oa))):.2e}, max rel Gamma diff = {dg_path:.2e}, same selected = {ref['path']['selected'] == other['path']['selected']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("narrative_ipca").setLevel(logging.INFO if args.verbose else logging.WARNING)

    if args.backends:
        backends = [s.strip() for s in args.backends.split(",") if s.strip()]
    else:
        backends = ["numba", "numpy"] if grouplasso.HAVE_NUMBA else ["numpy"]
    if args.no_fallback:
        backends = [b for b in backends if b != "numpy"]
    for b in backends:
        if b not in ("numba", "numpy"):
            print(f"unknown backend {b!r}; use numba,numpy", file=sys.stderr)
            return 2
        if b == "numba" and not grouplasso.HAVE_NUMBA:
            print("numba requested but not importable (install narrative-ipca[accel] or unset NARRATIVE_IPCA_NO_NUMBA)", file=sys.stderr)
            return 2
    if not backends:
        print("no backend left to run", file=sys.stderr)
        return 2

    pnl, build_s = build_full_panel(args.scenario, args.seed)
    cfg = EstimationConfig()
    results = [run_backend(b, pnl, cfg, args) for b in backends]
    print_tables(results, pnl, build_s, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
