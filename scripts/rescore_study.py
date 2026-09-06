"""Rescore a finished study with the current check set without re-running the pipeline (DESIGN.md D52).

``scripts/run_full_study.py`` stores every metric of every (scenario, seed)
run in ``<out>/<variant>/per_run.csv``. When the check set changes (which
metrics are pass/fail, which scenario gets which check, the thresholds), the
flags can be recomputed from those stored metrics. For every
``<out>/<variant>/per_run.csv`` this script

1. reads the rows (floats parsed exactly) and restores every metric value
   from the run's ``artefacts/<scenario>_seed<k>_metrics.json`` when that
   file exists - the artefact is the primary, bit-exact record of the run,
   so a value that drifted through a CSV round trip is put back;
2. recomputes every ``pass_<check>`` column with
   :func:`narrative_ipca.harness.evaluate_checks` (``nan`` where the check
   does not apply to the scenario; an erroring run keeps ``0.0`` on its
   applicable checks) and ``all_passed`` (vacuously ``True`` for a
   report-only scenario);
3. rebuilds ``summary.csv`` and ``passed.csv`` exactly as
   ``run_full_study.assemble`` does (that function is reused) and overwrites
   ``per_run.csv``;
4. removes the variant's ``harness_<date>.md`` and re-renders it with
   :func:`narrative_ipca.harness.write_report`; the report's setup section
   says ``rescored on <date> with check set <version>``.

The metrics themselves and the ``artefacts/`` directory are not touched. The
wall-clock timings of the setup section are read from ``<out>/study_run.log``
when it is present (``nan`` otherwise).

Usage (from the repository root):

    python scripts/rescore_study.py --out reports/simulation/full
    python scripts/rescore_study.py --out reports/simulation/full --variants bks,tol02
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for _p in (str(ROOT), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from narrative_ipca import harness as H  # noqa: E402
from narrative_ipca.config import HarnessThresholds, PipelineConfig  # noqa: E402
from narrative_ipca.simulation import scenario_config  # noqa: E402
from narrative_ipca.types import HarnessResult  # noqa: E402
from run_full_study import assemble, study_pipeline_config  # noqa: E402

logger = logging.getLogger("rescore_study")

REPORT_PATTERN = re.compile(r"^harness_\d{4}-\d{2}-\d{2}(_\d+)?\.md$")
"""File names of the harness reports a variant directory holds (``harness_<date>.md``, ``harness_<date>_2.md``, ...)."""

_LOG_STAMP = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+"
_LOG_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------
def load_rows(per_run: Path) -> list[dict[str, Any]]:
    """The rows of ``per_run.csv`` as dicts in file order (``error`` is ``""`` when empty, ``seed`` an ``int``).

    Floats are parsed exactly (``float_precision="round_trip"``): pandas'
    default parser is off by an ulp on a share of the 17-digit values, which
    would drift the stored metrics at every rescoring.
    """
    df = pd.read_csv(per_run, float_precision="round_trip")
    if "scenario" not in df.columns or "seed" not in df.columns:
        raise ValueError(f"{per_run}: needs 'scenario' and 'seed' columns")
    if "error" in df.columns:
        df["error"] = df["error"].fillna("").astype(str)
    else:
        df["error"] = ""
    rows = df.to_dict("records")
    for r in rows:
        r["scenario"] = str(r["scenario"])
        r["seed"] = int(r["seed"])
    return rows


def refresh_metrics_from_artefact(row: dict[str, Any], metrics_json: Path) -> int:
    """Overwrite the metric values of ``row`` with the ``values`` of its ``<run>_metrics.json``; returns the count.

    The artefact is the primary record of a run: ``run_full_study.run_one``
    writes it from the same ``HarnessMetrics.values`` the row was built from,
    and JSON keeps every float exactly, so a value that drifted through a CSV
    round trip is restored. Only numeric keys that are columns of the row are
    touched (never ``scenario`` / ``seed``); a missing file, an unreadable one
    or one without ``values`` leaves the row unchanged.
    """
    if not metrics_json.is_file():
        return 0
    try:
        values = json.loads(metrics_json.read_text(encoding="utf-8")).get("values")
    except (ValueError, AttributeError):
        return 0
    if not isinstance(values, dict):
        return 0
    n = 0
    for metric, value in values.items():
        if metric in ("scenario", "seed"):
            continue
        if metric in row and isinstance(value, (int, float)) and not isinstance(value, bool):
            row[metric] = float(value)
            n += 1
    return n


def rescore_rows(rows: list[dict[str, Any]], thresholds: HarnessThresholds) -> list[dict[str, Any]]:
    """Recompute ``pass_<check>`` and ``all_passed`` of every row in place from its metric values.

    ``pass_<check>`` is ``1.0`` / ``0.0`` for the checks of
    :func:`narrative_ipca.harness.scenario_checks` and ``nan`` for the others
    (every check of a report-only scenario, every metric of
    :data:`narrative_ipca.harness.REPORT_ONLY_METRICS`); a row with an
    ``error`` keeps ``0.0`` on its applicable checks and ``all_passed =
    False``, as ``run_full_study.run_one`` records it. ``all_passed`` is
    vacuously ``True`` for a report-only scenario that ran without error.
    """
    checks_all = [name for name, *_ in H.CHECKS]
    for r in rows:
        scenario = r["scenario"]
        applicable = H.scenario_checks(scenario)
        if str(r.get("error") or ""):
            flags = {c: (0.0 if c in applicable else float("nan")) for c in checks_all}
            all_passed = False
        else:
            passed = H.evaluate_checks(r, thresholds, scenario)
            flags = {c: (float(passed[c]) if c in passed else float("nan")) for c in checks_all}
            all_passed = all(passed.values())
        for c in checks_all:
            r[f"pass_{c}"] = flags[c]
        r["all_passed"] = bool(all_passed)
    return rows


# ---------------------------------------------------------------------------
# metadata of the original run
# ---------------------------------------------------------------------------
def timings_from_log(out: Path, variant: str) -> tuple[datetime | None, float]:
    """``(started, elapsed_seconds)`` of ``variant`` read from ``<out>/study_run.log``; ``(None, nan)`` when absent.

    The start is the driver's ``variant <name>: N runs`` line, the end the
    ``write_report`` line whose path lies in the variant's directory.
    """
    log = out / "study_run.log"
    if not log.is_file():
        return None, float("nan")
    start_re = re.compile(_LOG_STAMP + rf" run_full_study INFO variant {re.escape(variant)}: \d+ runs")
    end_re = re.compile(_LOG_STAMP + r" narrative_ipca\.harness INFO write_report: (.*)$")
    started: datetime | None = None
    finished: datetime | None = None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if started is None:
            m = start_re.search(line)
            if m:
                started = datetime.strptime(m.group(1), _LOG_FORMAT)
            continue
        m = end_re.search(line)
        if m:
            path = m.group(2).strip().replace("\\", "/")  # group 1 is the timestamp
            if f"/{variant}/" in path or path.startswith(f"{variant}/"):
                finished = datetime.strptime(m.group(1), _LOG_FORMAT)
                break
    if started is None:
        return None, float("nan")
    elapsed = (finished - started).total_seconds() if finished is not None else float("nan")
    return started, elapsed


def pipeline_config_of(variant: str) -> PipelineConfig:
    """The study's pipeline config of ``variant``; the full-size default for a directory that is not a study variant."""
    try:
        return study_pipeline_config(variant)
    except KeyError:
        logger.warning("variant %r is not a study tuning variant; the report shows the full-size default pipeline config", variant)
        return replace(H.default_pipeline_config(fast=False), name=f"study-{variant}")


def artefact_files(art: Path, scenario: str, seed: int) -> dict[str, str]:
    """The per-run artefact paths that exist under ``art`` (as ``run_full_study.run_one`` records them)."""
    key = f"{scenario}_seed{seed}"
    candidates = {
        "metrics": art / f"{key}_metrics.json",
        "lambda_path": art / f"{key}_lambda_path.csv",
        "gamma_norms": art / f"{key}_gamma_norms.csv",
    }
    return {name: str(p) for name, p in candidates.items() if p.is_file()}


# ---------------------------------------------------------------------------
# one variant
# ---------------------------------------------------------------------------
def rescore_variant(out_root: Path, variant: str, thresholds: HarnessThresholds | None = None) -> tuple[HarnessResult, str]:
    """Rescore ``<out_root>/<variant>``: rewrite the three tables, replace the report; returns ``(result, report path)``."""
    out = out_root / variant
    per_run = out / "per_run.csv"
    if not per_run.is_file():
        raise FileNotFoundError(per_run)
    thr = thresholds if thresholds is not None else HarnessThresholds()
    art = out / "artefacts"
    rows = load_rows(per_run)
    n_refreshed = sum(
        refresh_metrics_from_artefact(r, art / f"{r['scenario']}_seed{r['seed']}_metrics.json") > 0 for r in rows
    )
    rows = rescore_rows(rows, thr)
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    seeds = sorted({r["seed"] for r in rows})
    for r in rows:
        files = artefact_files(art, r["scenario"], r["seed"])
        if files:
            r["_files"] = files
        try:
            r["_sim_summary"] = H._sim_summary(replace(scenario_config(r["scenario"]), seed=r["seed"]))
        except (KeyError, ValueError):  # a scenario name the simulation module does not know
            pass
    started, elapsed = timings_from_log(out_root, variant)
    pcfg = pipeline_config_of(variant)
    result = assemble(rows, scenarios, seeds, pcfg, thr, out, started or datetime.now(timezone.utc), elapsed)
    result.meta["rescored"] = True
    result.meta["rescored_at"] = datetime.now(timezone.utc).isoformat()
    result.meta["check_set"] = H.CHECK_SET_VERSION
    result.meta["timings_from_log"] = started is not None
    removed = []
    for old in sorted(out.glob("harness_*.md")):
        if REPORT_PATTERN.match(old.name):
            old.unlink()
            removed.append(old.name)
    report = H.write_report(result, out)
    logger.info(
        "variant %s: %d run(s) rescored with check set %s (%d with metric values restored from the artefacts); "
        "removed %s; wrote %s",
        variant, len(rows), H.CHECK_SET_VERSION, n_refreshed, ", ".join(removed) or "no old report", report,
    )
    return result, report


def find_variants(out_root: Path) -> list[str]:
    """The sub-directories of ``out_root`` that hold a ``per_run.csv``, sorted by name."""
    if not out_root.is_dir():
        return []
    return sorted(d.name for d in out_root.iterdir() if d.is_dir() and (d / "per_run.csv").is_file())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/simulation/full", help="study directory holding <variant>/per_run.csv")
    ap.add_argument("--variants", default=None, help="comma-separated variant directories (default: every one with a per_run.csv)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    out_root = Path(args.out)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()] if args.variants else find_variants(out_root)
    if not variants:
        raise SystemExit(f"no <variant>/per_run.csv under {out_root}")
    for variant in variants:
        result, report = rescore_variant(out_root, variant)
        print(f"\n=== variant {variant}: rescored with check set {H.CHECK_SET_VERSION} (DESIGN.md D52), report {report}")
        print(f"reported only (no pass flag): {', '.join(H.REPORT_ONLY_METRICS)}")
        with pd.option_context("display.width", 200, "display.max_columns", 40):
            print(result.passed.dropna(axis=1, how="all").round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
