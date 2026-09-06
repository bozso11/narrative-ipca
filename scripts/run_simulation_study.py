"""Run the simulation harness and write the Part E report (DESIGN.md Part E).

Usage (from the repository root, no installation needed)::

    python scripts/run_simulation_study.py                            # HarnessConfig.scenarios, 3 seeds
    python scripts/run_simulation_study.py --scenarios baseline,no_factor,topic_null --seeds 3 --out reports/simulation
    python scripts/run_simulation_study.py --fast                     # reduced sizes, quick smoke run
    python scripts/run_simulation_study.py --config configs/default.yaml --seeds 1

The script calls :func:`narrative_ipca.harness.run_harness` and
:func:`narrative_ipca.harness.write_report`, prints the report path and the
pass/fail table, and exits with ``0`` when every applicable check passed in
every run, ``1`` otherwise (``2`` on a usage or data error), so that it can
serve as a regression gate. The report-only scenario ``topic_null`` has no
check and never fails the gate (unless its run raises).

The two null scenarios (see ``narrative_ipca.simulation`` and ``harness``):

* ``no_factor`` - the chance-level null: no topic carries information *and*
  returns have no common factor structure (every loading is zero), so the
  kernel covariances carry no information; selection is expected at chance
  and the OOS Sharpe within two standard errors of zero (pass/fail checks
  ``null_selection_lift``, ``null_oos_sharpe_abs``; the placebo count is
  reported against its chance level ``n_placebo / L``, not checked, D52).
* ``topic_null`` (alias ``null``) - no topic carries information but returns
  keep their priced factor structure; the instruments of pure noise topics
  then span the true loadings (their kernel covariances are ``beta_i' G_{t,l}``
  with a persistent common vector ``G``), IPCA recovers the factors and earns
  the premium. Selection above chance and a positive OOS Sharpe are the
  estimator's correct behaviour, so the scenario is reported (instrument R2
  on the true loadings vs chance, selection stability, OOS Sharpe vs
  baseline), not pass/failed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from narrative_ipca.config import HarnessConfig, PipelineConfig, SimulationConfig, load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    default_scenarios = ",".join(HarnessConfig().scenarios)
    parser = argparse.ArgumentParser(
        prog="run_simulation_study",
        description="Simulation harness of narrative-ipca: scenarios x seeds, comparison with the truth, markdown report.",
        epilog=(
            "Two null scenarios: no_factor is the chance-level null (no topic information and no common factor "
            "structure in returns; tested for selection at chance and an OOS Sharpe near zero); topic_null (alias "
            "null) removes the topic information but keeps the priced factor structure, under which the instruments "
            "of pure noise topics span the true loadings and the estimator legitimately earns a positive OOS Sharpe - "
            "it is report-only (no pass/fail check) and shows why an OOS Sharpe alone cannot certify narrative "
            "information."
        ),
    )
    parser.add_argument(
        "--scenarios", default=None,
        help=(
            "comma-separated scenario names among baseline | no_factor (chance-level null) | topic_null (alias null; "
            f"report-only, priced factor structure kept) | softmax | weak | balanced; default: {default_scenarios} "
            "(HarnessConfig.scenarios)"
        ),
    )
    parser.add_argument("--seeds", type=int, default=None, help="number of seeds per scenario (default: HarnessConfig.n_seeds)")
    parser.add_argument("--fast", action="store_true", help="reduced sizes (150 assets, 40 topics, 8 years) and the fast pipeline config")
    parser.add_argument("--out", type=Path, default=None, help="report directory (default: HarnessConfig.output_dir)")
    parser.add_argument("--config", type=Path, default=None, help="PipelineConfig JSON/YAML used for every run (default: harness defaults)")
    parser.add_argument("--base", type=Path, default=None, help="SimulationConfig JSON/YAML fixing the sizes of every scenario")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging (default INFO)")
    parser.add_argument("-q", "--quiet", action="store_true", help="WARNING logging only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("narrative_ipca").setLevel(level)

    from narrative_ipca import harness  # heavy import after logging is configured

    hcfg = HarnessConfig()
    updates: dict = {}
    if args.scenarios:
        updates["scenarios"] = tuple(s.strip() for s in str(args.scenarios).split(",") if s.strip())
    if args.seeds is not None:
        updates["n_seeds"] = int(args.seeds)
    if args.fast:
        updates["fast"] = True
    if args.out is not None:
        updates["output_dir"] = str(args.out)
    if updates:
        hcfg = replace(hcfg, **updates)
    try:
        pipeline_cfg = load_config(args.config, PipelineConfig) if args.config is not None else None
        base = load_config(args.base, SimulationConfig) if args.base is not None else None
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        result = harness.run_harness(hcfg, pipeline_cfg, base=base, progress=_progress)
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = harness.write_report(result, hcfg.output_dir)
    print(f"report: {report}")
    print()
    print("pass/fail (share of seeds passing each applicable check; blank = not applicable):")
    with_pct = result.passed.copy()
    print(with_pct.to_string(float_format=lambda v: f"{v:.2f}", na_rep=""))
    if result.meta.get("errors"):
        print()
        print("failed runs: " + "; ".join(f"{k}: {v}" for k, v in result.meta["errors"].items()))
    all_ok = bool(result.per_run["all_passed"].astype(bool).all()) and not result.meta.get("errors")
    print()
    print("all checks passed" if all_ok else "some checks failed (see the report)")
    return 0 if all_ok else 1


def _progress(done: int, total: int, message: str) -> None:
    print(f"[{done}/{total}] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
