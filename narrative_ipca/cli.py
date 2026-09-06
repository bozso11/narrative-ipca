"""Command-line interface (DESIGN.md Part C, ``pipeline.py`` / ``cli.py``).

Three sub-commands::

    narrative-ipca run      --config cfg.yaml --attention a.parquet --returns r.parquet [--meta m.csv] --out dir
    narrative-ipca simulate [--config sim.yaml] [--scenario baseline] [--seed 0] [--fast] --out dir
    narrative-ipca harness  --scenarios baseline,no_factor,topic_null --seeds 3 [--fast] --out reports/simulation

Scenario names (``simulate --scenario``, ``harness --scenarios``):
``baseline | no_factor | topic_null (alias null) | softmax | weak | balanced``,
each optionally with a ``-fast`` suffix (see :func:`narrative_ipca.simulation.scenario_config`
and DESIGN.md Part D). ``no_factor`` is the chance-level null (no topic
information, no common factor structure in returns); ``topic_null`` keeps
the priced factor structure and is report-only in the harness (D47).

``run`` loads the inputs (:func:`narrative_ipca.pipeline.load_inputs`), runs
:func:`narrative_ipca.pipeline.run_pipeline` and writes the artefacts with
:func:`narrative_ipca.pipeline.save_result`. ``simulate`` draws one data set
from :func:`narrative_ipca.simulation.simulate` and writes it in the input
format ``run`` reads (``attention.parquet`` -- or ``.csv`` without pyarrow --,
``returns.csv``, ``asset_meta.csv``), plus ``truth_summary.json``,
``truth.npz`` and ``simulation_config.json``. ``harness`` delegates to
:mod:`narrative_ipca.harness` (imported lazily, with a clear error when the
module is not available).

Logging goes to stderr at ``INFO`` (``--verbose``: ``DEBUG``) through the
standard ``logging`` module (D44). :func:`main` returns the process exit
code: ``0`` on success, ``1`` on a data/config error, ``2`` on a usage
error or a missing optional module.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import __version__
from .config import HarnessConfig, PipelineConfig, SimulationConfig, load_config, save_config

logger = logging.getLogger(__name__)

__all__ = ["main", "build_parser"]

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """The argparse parser of the ``narrative-ipca`` command."""
    parser = argparse.ArgumentParser(
        prog="narrative-ipca",
        description="Narrative asset pricing (Bybee, Kelly & Su 2023): attention shocks -> narrative "
        "covariances -> Sparse IPCA -> narrative factors.",
    )
    parser.add_argument("--version", action="version", version=f"narrative-ipca {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging (default INFO)")
    parser.add_argument("-q", "--quiet", action="store_true", help="WARNING logging only")
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    run = sub.add_parser("run", help="run the pipeline on attention and return panels")
    run.add_argument("--config", type=Path, default=None, help="PipelineConfig JSON/YAML (default: package defaults)")
    run.add_argument("--attention", type=Path, required=True, help="dates x topics attention levels (.parquet/.csv)")
    run.add_argument("--returns", type=Path, required=True, help="dates x assets daily returns (.parquet/.csv)")
    run.add_argument("--meta", type=Path, default=None, help="asset metadata table indexed by asset id")
    run.add_argument("--risk-free", type=Path, default=None, help="daily risk-free series (return_kind='total')")
    run.add_argument("--test-assets", type=Path, default=None, help="dates x test-asset excess returns for the pricing tests")
    run.add_argument(
        "--observables", type=Path, default=None,
        help="dates x observable series (e.g. a market factor) projected on the factors and used as benchmark factors",
    )
    run.add_argument("--out", type=Path, default=None, help="output directory (default: PipelineConfig.output_dir)")
    run.add_argument("--save-panel", action="store_true", help="also write the covariance panel and the long-form panel")
    run.add_argument("--name", default=None, help="run name recorded in the config")
    run.set_defaults(func=_cmd_run)

    sim = sub.add_parser("simulate", help="write a simulated data set with known ground truth")
    sim.add_argument("--config", type=Path, default=None, help="SimulationConfig JSON/YAML (default: package defaults)")
    sim.add_argument(
        "--scenario", default=None,
        help="baseline | no_factor | topic_null (alias null) | softmax | weak | balanced (optionally with a '-fast' suffix)",
    )
    sim.add_argument("--seed", type=int, default=None, help="override the seed")
    sim.add_argument("--fast", action="store_true", help="reduced sizes (150 assets, 40 topics, 8 years)")
    sim.add_argument("--csv", action="store_true", help="write attention.csv instead of attention.parquet")
    sim.add_argument("--out", type=Path, required=True, help="output directory")
    sim.set_defaults(func=_cmd_simulate)

    har = sub.add_parser("harness", help="run the simulation harness (needs narrative_ipca.harness)")
    har.add_argument(
        "--scenarios", default=None,
        help="comma-separated scenario names among baseline | no_factor | topic_null (alias null) | softmax | weak | "
        "balanced (default: HarnessConfig.scenarios)",
    )
    har.add_argument("--seeds", type=int, default=None, help="number of seeds per scenario")
    har.add_argument("--fast", action="store_true", help="fast (reduced-size) variant of every scenario")
    har.add_argument("--config", type=Path, default=None, help="PipelineConfig JSON/YAML used for every run")
    har.add_argument("--out", type=Path, default=None, help="report directory (default: HarnessConfig.output_dir)")
    har.set_defaults(func=_cmd_harness)
    return parser


def _configure_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(level=level, format=_LOG_FORMAT, stream=sys.stderr)
    logging.getLogger("narrative_ipca").setLevel(level)


def _load_pipeline_config(path: Path | None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    cfg = load_config(path, PipelineConfig)
    if not isinstance(cfg, PipelineConfig):  # pragma: no cover - load_config builds the requested class
        raise TypeError(f"{path} did not produce a PipelineConfig")
    return cfg


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def _cmd_run(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from .pipeline import load_inputs, read_frame, run_pipeline, save_result

    cfg = _load_pipeline_config(args.config)
    updates: dict[str, Any] = {}
    if args.out is not None:
        updates["output_dir"] = str(args.out)
    if args.save_panel:
        updates["save_panel"] = True
    if args.name:
        updates["name"] = str(args.name)
    if updates:
        cfg = replace(cfg, **updates)
    if not cfg.output_dir:
        logger.error("no output directory: pass --out or set PipelineConfig.output_dir")
        return 2
    out = Path(cfg.output_dir)

    attention, returns = load_inputs(args.attention, args.returns, meta_path=args.meta, risk_free_path=args.risk_free)
    test_assets = read_frame(args.test_assets, dates=True) if args.test_assets is not None else None
    observables: dict[str, pd.Series] | None = None
    if args.observables is not None:
        obs_frame = read_frame(args.observables, dates=True)
        observables = {str(c): obs_frame[c].astype(float) for c in obs_frame.columns}

    logger.info("run %r: config hash %s, output %s", cfg.name, cfg.hash(), out)
    result = run_pipeline(attention, returns, cfg, test_assets=test_assets, observables=observables)
    manifest = save_result(result, out)
    m = result.evaluation.metrics
    print(f"narrative-ipca run {cfg.name!r} finished in {result.timings.get('total', float('nan')):.1f}s -> {out}")
    print(
        f"  lambda*={m.get('lam_star', float('nan')):.4g}  K={int(m.get('K', 0))}  "
        f"selected={int(m.get('n_selected', 0))}/{int(m.get('L', 0))}  "
        f"total_r2={m.get('total_r2', float('nan')):.4f}  IS Sharpe={m.get('mve_sharpe_is', float('nan')):.3f}"
        + (f"  OOS Sharpe={m['oos_sharpe']:.3f}" if "oos_sharpe" in m else "")
    )
    print(f"  {len(manifest)} files, e.g. {Path(manifest['metrics']).name}, {Path(manifest['gamma']).name}")
    return 0


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------
def _cmd_simulate(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from .simulation import scenario_config, simulate

    if args.config is not None:
        cfg = load_config(args.config, SimulationConfig)
    else:
        cfg = SimulationConfig()
    if args.scenario is not None or args.fast:
        cfg = scenario_config(args.scenario or "baseline", base=cfg, fast=bool(args.fast))
    if args.seed is not None:
        cfg = replace(cfg, seed=int(args.seed))

    sim = simulate(cfg, scenario=args.scenario)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = _write_simulation(sim, out, csv=bool(args.csv))
    truth = sim.truth
    print(
        f"simulated scenario={truth.meta.get('scenario')} seed={cfg.seed}: {sim.attention.levels.shape[0]} days, "
        f"{len(truth.periods)} periods, {sim.returns.returns.shape[1]} assets, {sim.attention.levels.shape[1]} topics "
        f"(true MVE Sharpe {truth.sharpe_mve_true:.2f}) -> {out}"
    )
    for name in written:
        print(f"  {name}")
    return 0


def _write_simulation(sim: Any, out: Path, csv: bool = False) -> list[str]:
    """Write a :class:`SimulatedData` set in the input format of ``run``; returns the file names."""
    written: list[str] = []
    att = sim.attention.levels.rename_axis("date")
    if not csv:
        try:
            att.to_parquet(out / "attention.parquet")
            written.append("attention.parquet")
        except (ImportError, ValueError) as exc:
            logger.warning("attention.parquet not written (%s); writing attention.csv", exc)
            csv = True
    if csv:
        att.to_csv(out / "attention.csv")
        written.append("attention.csv")
    sim.returns.returns.rename_axis("date").to_csv(out / "returns.csv")
    written.append("returns.csv")
    if sim.returns.asset_meta is not None:
        sim.returns.asset_meta.rename_axis("asset").to_csv(out / "asset_meta.csv")
        written.append("asset_meta.csv")
    truth = sim.truth
    topics = list(sim.attention.topics)
    summary = {
        "scenario": truth.meta.get("scenario"),
        "seed": int(sim.config.seed),
        "n_days": int(sim.attention.levels.shape[0]),
        "n_periods": int(len(truth.periods)),
        "n_assets": int(sim.returns.returns.shape[1]),
        "n_topics": int(len(topics)),
        "K": int(truth.A.shape[1]),
        "relevant_topics": [t for t, r in zip(topics, truth.relevant) if r],
        "placebo_topics": [t for t, p in zip(topics, truth.placebo) if p],
        "sharpe_mve_true": float(truth.sharpe_mve_true),
        "sharpe_mve_realized": truth.meta.get("sharpe_mve_realized"),
        "systematic_r2": float(truth.systematic_r2),
        "mu_f_period": np.asarray(truth.mu_f_period).tolist(),
        "Sigma_ff_period_diag": np.diag(np.asarray(truth.Sigma_ff_period)).tolist(),
        "periods_per_year": truth.meta.get("periods_per_year"),
        "class_counts": truth.meta.get("class_counts"),
        "observed_fraction": truth.meta.get("observed_fraction"),
    }
    (out / "truth_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    written.append("truth_summary.json")
    np.savez(
        out / "truth.npz",
        A=np.asarray(truth.A),
        relevant=np.asarray(truth.relevant),
        placebo=np.asarray(truth.placebo),
        Gamma_tilde_true=np.asarray(truth.Gamma_tilde_true),
        impact_z_to_mve_true=np.asarray(truth.impact_z_to_mve_true),
        mu_f_period=np.asarray(truth.mu_f_period),
        Sigma_ff_period=np.asarray(truth.Sigma_ff_period),
        Sigma_ff_daily=np.asarray(truth.Sigma_ff_daily),
        f_period=truth.f_period.to_numpy(),
        periods=np.asarray(pd.DatetimeIndex(truth.periods).asi8),
        topics=np.asarray(topics, dtype=str),
    )
    written.append("truth.npz")
    truth.f_period.rename_axis("period").to_csv(out / "truth_f_period.csv")
    written.append("truth_f_period.csv")
    save_config(sim.config, out / "simulation_config.json")
    written.append("simulation_config.json")
    logger.info("simulate: wrote %s to %s", ", ".join(written), out)
    return written


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------
def _cmd_harness(args: argparse.Namespace) -> int:
    from dataclasses import replace

    try:
        from . import harness as _harness
    except ImportError as exc:
        logger.error(
            "the simulation harness is not available (narrative_ipca.harness could not be imported: %s); "
            "install the full package or run 'narrative-ipca simulate' + 'narrative-ipca run' instead", exc,
        )
        print("error: narrative_ipca.harness is not available in this installation", file=sys.stderr)
        return 2

    hcfg = HarnessConfig()
    updates: dict[str, Any] = {}
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
    pipeline_cfg = _load_pipeline_config(args.config) if args.config is not None else None

    logger.info("harness: scenarios=%s seeds=%d fast=%s out=%s", hcfg.scenarios, hcfg.n_seeds, hcfg.fast, hcfg.output_dir)
    result = _harness.run_harness(hcfg, pipeline_cfg)
    report = _harness.write_report(result, hcfg.output_dir)
    print(f"harness report: {report}")
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` (default ``sys.argv[1:]``), run the sub-command and return the exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _configure_logging(bool(args.verbose), bool(args.quiet))
    try:
        return int(args.func(args))
    except (ValueError, TypeError, FileNotFoundError, KeyError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
