"""Cross-variant summary of a full study run (see scripts/run_full_study.py).

Reads ``<out>/<variant>/per_run.csv`` for every variant found under ``<out>``
and writes ``<out>/STUDY_<date>.md`` with, per scenario, the key metrics of
every tuning variant side by side (mean and standard deviation over seeds)
and the pass rates of the applicable checks. The per-variant harness reports
(``<out>/<variant>/harness_<date>.md``) keep the full detail.

Usage: python scripts/summarise_study.py --out reports/simulation/full
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from narrative_ipca.harness import CHECK_SET_VERSION, REPORT_ONLY_METRICS  # noqa: E402

KEY_METRICS = [
    ("n_selected", "selected"),
    ("selection_recall", "recall"),
    ("selection_recall_strong", "recall (strong)"),
    ("selection_precision", "precision"),
    ("placebo_selected", "placebos selected"),
    ("mve_sharpe_is", "IS Sharpe"),
    ("oos_sharpe", "OOS Sharpe"),
    ("oos_sharpe_true_mve", "OOS Sharpe true MVE"),
    ("oos_sharpe_ratio_to_true", "OOS ratio"),
    ("beta_canonical_corr", "beta CC1"),
    ("factor_canonical_corr", "factor CC1"),
    ("factor_canonical_corr_mean", "factor CC mean"),
    ("state_canonical_corr", "state CC1"),
    ("gamma_subspace_cos", "Gamma cos (selected rows)"),
    ("impact_spearman", "impact rho (selected rows)"),
    ("systematic_r2_recovered", "systematic R2 recovered"),
    ("total_r2", "total R2"),
    ("instrument_beta_r2_relevant", "instr. R2 relevant"),
    ("instrument_beta_r2_noise", "instr. R2 noise"),
    ("instrument_beta_r2_chance", "instr. R2 chance"),
    ("oos_selection_stability", "selection stability"),
    ("null_selection_lift", "selection lift vs 5% chance"),
    ("lam_star", "lambda*"),
    ("runtime_seconds", "runtime s"),
]


def _fmt(m: float, s: float, n: int) -> str:
    if not np.isfinite(m):
        return "n/a"
    if n <= 1 or not np.isfinite(s):
        return f"{m:.3g}"
    return f"{m:.3g} ± {s:.2g}"


def load(out: Path) -> dict[str, pd.DataFrame]:
    runs = {}
    for d in sorted(out.iterdir()):
        f = d / "per_run.csv"
        if d.is_dir() and f.exists():
            runs[d.name] = pd.read_csv(f)
    if not runs:
        raise SystemExit(f"no <variant>/per_run.csv under {out}")
    return runs


def build(runs: dict[str, pd.DataFrame]) -> str:
    variants = list(runs)
    scenarios: list[str] = []
    for df in runs.values():
        for s in df["scenario"].tolist():
            if s not in scenarios:
                scenarios.append(s)
    parts = [f"# Simulation study summary ({datetime.now():%Y-%m-%d})", ""]
    parts.append(
        "Tuning variants: `bks` = BKS in-sample MVE Sharpe, exact argmax; `tol02` = same criterion, sparsest point within 2% of the maximum; "
        "`loocv` = leave-one-period-out Sharpe (BKS App. C.3) on 16 subsampled folds. Values are mean ± std over seeds. "
        "Metric definitions: DESIGN.md Part E."
    )
    parts.append("")
    parts.append(
        f"Check set {CHECK_SET_VERSION} (DESIGN.md D52): {', '.join(f'`{m}`' for m in REPORT_ONLY_METRICS)} are reported "
        "only (not identified targets of the model: Gamma_tilde is identified only up to an (L - K)-dimensional family); "
        "the recall checks are soft (a sparse representative may use only the strong topics); under `no_factor` the "
        "placebo count is reported against its chance level n_placebo / L, not checked."
    )
    parts.append("")
    for s in scenarios:
        parts.append(f"## {s}")
        parts.append("")
        header = "| metric | " + " | ".join(variants) + " |"
        parts.append(header)
        parts.append("|---|" + "---|" * len(variants))
        for col, label in KEY_METRICS:
            if col in REPORT_ONLY_METRICS:
                label = f"{label}; reported only"
            cells = []
            for v in variants:
                df = runs[v]
                sub = df[(df["scenario"] == s)]
                if col not in sub or sub.empty:
                    cells.append("n/a")
                    continue
                x = pd.to_numeric(sub[col], errors="coerce")
                cells.append(_fmt(float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan"), int(x.notna().sum())))
            parts.append(f"| {label} | " + " | ".join(cells) + " |")
        # pass rates
        pass_cols = sorted({c for df in runs.values() for c in df.columns if c.startswith("pass_")})
        rows = []
        for c in pass_cols:
            cells = []
            any_applicable = False
            for v in variants:
                sub = runs[v][runs[v]["scenario"] == s]
                if c not in sub or sub.empty:
                    cells.append("n/a")
                    continue
                x = pd.to_numeric(sub[c], errors="coerce")
                if x.notna().sum() == 0:
                    cells.append("n/a")
                else:
                    any_applicable = True
                    cells.append(f"{int(x.sum())}/{int(x.notna().sum())}")
            if any_applicable:
                rows.append(f"| {c[5:]} | " + " | ".join(cells) + " |")
        if rows:
            parts.append("")
            parts.append("Checks passed (seeds passing / seeds):")
            parts.append("")
            parts.append("| check | " + " | ".join(variants) + " |")
            parts.append("|---|" + "---|" * len(variants))
            parts.extend(rows)
        else:
            parts.append("")
            parts.append("Report-only scenario (no pass/fail checks).")
        errs = {v: runs[v][(runs[v]["scenario"] == s) & (runs[v]["error"].fillna("") != "")] for v in variants}
        n_err = sum(len(e) for e in errs.values())
        if n_err:
            parts.append("")
            parts.append(f"Failed runs: {n_err} — " + "; ".join(f"{v}: {len(e)}" for v, e in errs.items() if len(e)))
        parts.append("")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/simulation/full")
    args = ap.parse_args(argv)
    out = Path(args.out)
    runs = load(out)
    text = build(runs)
    path = out / f"STUDY_{datetime.now():%Y-%m-%d}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print("\nwritten:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
