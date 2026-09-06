"""Full-size simulation study, parallelised over (scenario, seed) runs.

``narrative_ipca.harness.run_harness`` runs its scenario x seed grid
sequentially. This driver runs the same grid through a process pool (one
process per run, a few BLAS threads each) and assembles the identical
``HarnessResult`` so that ``harness.write_report`` produces the same report.
It also runs *tuning variants* of the baseline scenario side by side (BKS
exact argmax, the relative-tolerance rule, LOOCV), which is how the study
compares the lambda selection rules (DESIGN.md D27).

Usage (from the repository root):

    python scripts/run_full_study.py --scenarios baseline,no_factor,topic_null,softmax,weak --seeds 3 --workers 7
    python scripts/run_full_study.py --variants bks,tol02,loocv --variant-scenarios baseline --seeds 3

Outputs: ``<out>/<variant>/`` with ``per_run.csv``, ``summary.csv``,
``passed.csv``, ``artefacts/`` and the markdown report.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_THREADS = "3"
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, DEFAULT_THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from narrative_ipca import __version__  # noqa: E402
from narrative_ipca.config import (  # noqa: E402
    CovarianceConfig,
    EstimationConfig,
    HarnessConfig,
    HarnessThresholds,
    LambdaGridConfig,
    OOSConfig,
    PipelineConfig,
    SimulationConfig,
    TuningConfig,
    config_to_dict,
)
from narrative_ipca.types import HarnessResult  # noqa: E402

logger = logging.getLogger("run_full_study")


# ---------------------------------------------------------------------------
# Pipeline variants
# ---------------------------------------------------------------------------
def study_pipeline_config(variant: str = "bks") -> PipelineConfig:
    """The pipeline configuration of the study (full sizes) for one tuning variant.

    ``bks``: BKS defaults, in-sample Sharpe exact argmax; ``tol02``: the same
    with ``TuningConfig.tolerance = 0.02`` (sparsest point within 2% of the
    best Sharpe); ``loocv``: leave-one-period-out Sharpe on 16 subsampled
    folds (BKS App. C.3). The lambda grid is 20 log-spaced points over four
    decades below ``lam_max`` (D22): on this panel the in-sample Sharpe
    surface is flat down to about ``1e-3 lam_max`` and the exact argmax must
    not sit on the grid boundary, so the grid extends a decade further. The
    OOS window is the last 40% of periods with annual refits.
    """
    tuning = {
        "bks": TuningConfig(criterion="is_sharpe"),
        "tol02": TuningConfig(criterion="is_sharpe", tolerance=0.02),
        "loocv": TuningConfig(criterion="loocv_sharpe", loocv_max_folds=16),
    }[variant]
    return PipelineConfig(
        covariance=CovarianceConfig(burn_in_periods=12),
        estimation=EstimationConfig(K=3, lam_grid=LambdaGridConfig(n_lambdas=20, ratio=1e-4)),
        tuning=tuning,
        oos=OOSConfig(oos_fraction=0.4, refit_every=12, retune_lambda=True, min_train_periods=60),
        name=f"study-{variant}",
    )


# ---------------------------------------------------------------------------
# One run (executed in a worker process)
# ---------------------------------------------------------------------------
def run_one(scenario: str, seed: int, pipeline_cfg: PipelineConfig, thresholds: HarnessThresholds, art_dir: str) -> dict[str, Any]:
    """simulate -> run_pipeline -> compare_to_truth for one (scenario, seed); returns the per_run row."""
    from narrative_ipca import harness as H
    from narrative_ipca.pipeline import run_pipeline
    from narrative_ipca.simulation import scenario_config, simulate

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    key = f"{scenario}_seed{seed}"
    art = Path(art_dir)
    art.mkdir(parents=True, exist_ok=True)
    checks_all = [name for name, *_ in H.CHECKS]
    applicable = H.scenario_checks(scenario)
    sim_cfg = replace(scenario_config(scenario), seed=int(seed))
    row: dict[str, Any] = {"scenario": scenario, "seed": int(seed)}
    t0 = time.perf_counter()
    try:
        data = simulate(sim_cfg, scenario=scenario)
        res = run_pipeline(data.attention, data.returns, replace(pipeline_cfg, name=f"study-{key}"))
        m = H.compare_to_truth(res, data.truth, thresholds, scenario, asset_ids=data.returns.assets)
    except Exception as exc:  # keep the other runs alive
        import traceback

        row.update({metric: float("nan") for metric in H.METRICS})
        row.update({f"pass_{c}": (0.0 if c in applicable else float("nan")) for c in checks_all})
        row["all_passed"] = False
        row["runtime_seconds"] = time.perf_counter() - t0
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["_traceback"] = traceback.format_exc()
        row["_sim_summary"] = H._sim_summary(sim_cfg)
        return row
    elapsed = time.perf_counter() - t0
    row.update({metric: float(v) for metric, v in m.values.items()})
    row.update({f"pass_{c}": (float(m.passed[c]) if c in m.passed else float("nan")) for c in checks_all})
    row["all_passed"] = bool(m.all_passed)
    row["runtime_seconds"] = float(m.values.get("runtime_seconds", elapsed))
    row["harness_seconds"] = elapsed
    row["error"] = ""
    files: dict[str, str] = {}
    p_metrics = art / f"{key}_metrics.json"
    H._write_json(
        p_metrics,
        {
            "scenario": scenario,
            "seed": int(seed),
            "values": m.values,
            "passed": m.passed,
            "details": m.details,
            "pipeline_metrics": res.evaluation.metrics,
            "timings": res.timings,
            "simulation": H._sim_summary(sim_cfg),
            "selected_topics": res.fit.selected_topics,
            "oos_lam_history": [float(x) for x in res.oos.lam_history] if res.oos is not None else [],
            "oos_n_selected_history": [int(x) for x in res.oos.n_selected_history] if res.oos is not None else [],
        },
    )
    files["metrics"] = str(p_metrics)
    p_path = art / f"{key}_lambda_path.csv"
    res.tuning.path_frame().rename_axis("point").to_csv(p_path)
    files["lambda_path"] = str(p_path)
    p_norms = art / f"{key}_gamma_norms.csv"
    H._gamma_norm_frame(res.tuning, res.fit.instrument_names).rename_axis("point").to_csv(p_norms)
    files["gamma_norms"] = str(p_norms)
    row["_files"] = files
    row["_sim_summary"] = H._sim_summary(sim_cfg)
    return row


# ---------------------------------------------------------------------------
# Assemble a HarnessResult exactly as harness.run_harness does
# ---------------------------------------------------------------------------
def assemble(rows: list[dict[str, Any]], scenarios: list[str], seeds: list[int], pcfg: PipelineConfig,
             thresholds: HarnessThresholds, out: Path, started: datetime, elapsed_all: float) -> HarnessResult:
    from narrative_ipca import harness as H
    from narrative_ipca.simulation import scenario_config

    checks_all = [name for name, *_ in H.CHECKS]
    artefacts = {f"{r['scenario']}_seed{r['seed']}": r.pop("_files") for r in rows if "_files" in r}
    sims = {r["scenario"]: r.pop("_sim_summary") for r in rows if "_sim_summary" in r}
    errors = {f"{r['scenario']}_seed{r['seed']}": r["error"] for r in rows if r.get("error")}
    tracebacks = {f"{r['scenario']}_seed{r['seed']}": r.pop("_traceback") for r in rows if "_traceback" in r}
    order = {(s, k): i for i, (s, k) in enumerate((s, k) for s in scenarios for k in seeds)}
    rows = sorted(rows, key=lambda r: order[(r["scenario"], r["seed"])])
    per_run = pd.DataFrame(rows)
    metric_cols = [c for c in H.METRICS if c in per_run.columns]
    extra_cols = [
        c for c in per_run.columns
        if c not in ("scenario", "seed", "all_passed", "error", "harness_seconds")
        and not c.startswith("pass_") and c not in metric_cols
    ]
    grouped = per_run.groupby("scenario", sort=False)
    summary = grouped[metric_cols + extra_cols].agg(["mean", "std"])
    pass_cols = [f"pass_{c}" for c in checks_all]
    passed = grouped[pass_cols].mean()
    passed.columns = checks_all
    passed["all"] = grouped["all_passed"].apply(lambda s: float(np.mean(s.astype(bool))))
    per_run.to_csv(out / "per_run.csv", index=False)
    summary.to_csv(out / "summary.csv")
    passed.to_csv(out / "passed.csv")
    if tracebacks:
        (out / "tracebacks.json").write_text(json.dumps(tracebacks, indent=1), encoding="utf-8")
    meta: dict[str, Any] = {
        "package_version": __version__,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_all,
        "n_runs": len(rows),
        "n_failed": len(errors),
        "scenarios": scenarios,
        "seeds": seeds,
        "fast": False,
        "sizes": "full",
        "pipeline_config": config_to_dict(pcfg),
        "pipeline_name": pcfg.name,
        "base_config": None,
        "simulation": sims,
        "output_dir": str(out),
        "artefact_dir": str(out / "artefacts"),
        "artefacts": artefacts,
        "tables": {"per_run": str(out / "per_run.csv"), "summary": str(out / "summary.csv"), "passed": str(out / "passed.csv")},
        "errors": errors,
        "checks": {s: list(H.scenario_checks(s)) for s in scenarios},
        "report_only": [s for s in scenarios if not H.scenario_checks(s)],
        "thresholds_used": {s: {f.name: getattr(H.scenario_thresholds(thresholds, s), f.name) for f in fields(HarnessThresholds)} for s in scenarios},
        "parallel_workers": True,
    }
    scenario_cfgs = {s: scenario_config(s) for s in scenarios}
    return HarnessResult(per_run=per_run, summary=summary, passed=passed, scenario_configs=scenario_cfgs, thresholds=thresholds, meta=meta)


def run_variant(variant: str, scenarios: list[str], seeds: list[int], out_root: Path, workers: int) -> tuple[HarnessResult, str]:
    from narrative_ipca import harness as H

    pcfg = study_pipeline_config(variant)
    thresholds = HarnessThresholds()
    out = out_root / variant
    art = out / "artefacts"
    art.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    jobs = [(s, k) for s in scenarios for k in seeds]
    logger.info("variant %s: %d runs on %d workers (%s threads each)", variant, len(jobs), workers, os.environ["OPENBLAS_NUM_THREADS"])
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, s, k, pcfg, thresholds, str(art)): (s, k) for s, k in jobs}
        for fut in as_completed(futs):
            s, k = futs[fut]
            row = fut.result()
            rows.append(row)
            status = "ERROR " + row["error"] if row.get("error") else ("pass" if row["all_passed"] else "FAIL")
            logger.info("variant %s: %s seed %d done in %.0fs -> %s (n_selected %s, oos_sharpe %s)", variant, s, k,
                        row.get("harness_seconds", float("nan")), status, row.get("n_selected"), row.get("oos_sharpe"))
    result = assemble(rows, scenarios, seeds, pcfg, thresholds, out, started, time.perf_counter() - t0)
    report = H.write_report(result, out)
    return result, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default=",".join(HarnessConfig().scenarios), help="comma-separated scenarios for the main (bks) variant")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", default="bks", help="comma-separated tuning variants: bks, tol02, loocv")
    ap.add_argument("--variant-scenarios", default=None, help="scenarios run for the non-bks variants (default: same as --scenarios)")
    ap.add_argument("--workers", type=int, default=max(1, min(7, (os.cpu_count() or 4) // 3)))
    ap.add_argument("--out", default="reports/simulation/full")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    out_root = Path(args.out)
    seeds = list(range(int(args.seeds)))
    reports: dict[str, str] = {}
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        scen = args.scenarios if (variant == "bks" or args.variant_scenarios is None) else args.variant_scenarios
        scenarios = [s.strip() for s in scen.split(",") if s.strip()]
        result, report = run_variant(variant, scenarios, seeds, out_root, int(args.workers))
        reports[variant] = report
        print(f"\n=== variant {variant}: report {report}")
        with pd.option_context("display.width", 200, "display.max_columns", 40):
            print(result.passed.round(2).to_string())
            cols = [c for c in ("n_selected", "selection_recall", "selection_precision", "placebo_selected", "mve_sharpe_is",
                                "oos_sharpe", "oos_sharpe_true_mve", "factor_canonical_corr", "gamma_subspace_cos",
                                "instrument_beta_r2_relevant", "instrument_beta_r2_noise", "oos_selection_stability", "runtime_seconds")
                    if c in result.per_run.columns]
            print(result.per_run[["scenario", "seed"] + cols].round(3).to_string(index=False))
    print("\nreports:", json.dumps(reports, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
