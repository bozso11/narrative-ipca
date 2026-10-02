# Simulation reports

What is in this folder and how to regenerate it.

## Contents

- `full/STUDY_<date>.md` — cross-variant summary of the headline study: per
  scenario, the key metrics of the three lambda-tuning variants side by side
  (`bks` exact in-sample argmax, `tol02` sparsest point within 2% of the best
  in-sample Sharpe, `loocv` leave-one-period-out Sharpe) and the pass rates of
  the applicable checks.
- `full/<variant>/harness_<date>.md` — the full harness report of one
  variant: setup, per-scenario summary (mean ± std over seeds), pass/fail
  table, per-run table, instrument informativeness, timings, and an
  "expected vs observed" paragraph per scenario.
- `full/<variant>/per_run.csv`, `summary.csv`, `passed.csv` — the tables
  behind the report. `full/<variant>/artefacts/` (per-run metrics JSON, lambda
  paths, gamma norms) is regenerable and not committed.
- `full/study_run.log` — the driver log with per-run timings.
- `timing/*.log` — single full-size baseline runs used to calibrate the
  study: the unaccelerated pure-Python solver (19 minutes) and the numba
  kernel (14 seconds), including the lambda paths that motivated the wider
  grid and the tolerance rule (DESIGN.md D27, D46, D50).

## Regenerate

From the repository root, with the `dev` extras installed (numba recommended):

```bash
python scripts/run_full_study.py --variants bks,tol02,loocv --seeds 3 --workers 8 --out reports/simulation/full
```

```bash
python scripts/summarise_study.py --out reports/simulation/full
```

After a change of the check set (which metrics are pass/fail, which
scenario gets which check, the thresholds; DESIGN.md D52 introduced check set
v2) an existing study is rescored from its stored metrics without re-running
the pipeline, then summarised again:

```bash
python scripts/rescore_study.py --out reports/simulation/full
python scripts/summarise_study.py --out reports/simulation/full
```

Note: the per-run artefact JSON files (`<variant>/artefacts/<run>_metrics.json`)
keep the `passed` dicts of the run that wrote them: check set v1 for `bks` and
`loocv` (2026-09-06), check set v2 for the `tol02` re-run of 2026-10-02. The
authoritative pass flags are the `pass_*` columns of `per_run.csv` and the
`passed.csv` table.

To re-run one variant (as `tol02` was on 2026-10-02 after the D51 fix), append
its log, then summarise again; `rescore_study.py` reads the timings of the
latest completed run of each variant from the log, so one log must hold one
run per variant at a time:

```bash
python scripts/run_full_study.py --variants tol02 --seeds 3 --workers 8 --out reports/simulation/full >> reports/simulation/full/study_run.log 2>&1
python scripts/summarise_study.py --out reports/simulation/full
```

The sequential single-process equivalent for one variant is
`python scripts/run_simulation_study.py --scenarios baseline,no_factor,topic_null,softmax,weak --seeds 3`
(BKS argmax only). Scenario definitions: DESIGN.md Part D; metric definitions
and thresholds: DESIGN.md Part E and `HarnessThresholds` in
`narrative_ipca/config.py`.
