"""Run the topic-sensitivity lab once from the command line (DESIGN.md G.9, G.13).

Runs every lab stage for one configuration with
:func:`narrative_ipca.exposure_lab.session.run_lab` and writes the results:

* ``oos_corr.csv``: OOS correlation, assets x topics (the correlation table
  of G.9, no blank rule);
* ``sensitivities.csv``: per topic-asset pair (long form, standardised units)
  the estimated sensitivity ``b_hat``, the true sensitivity ``b_true``, the set
  sensitivity ``w_set`` and whether the estimator selected the pair;
* ``r2.csv``: per asset OOS R2 of the estimator and the oracle, population R2,
  realised, explained and residual move over the forecast window;
* ``contributions.csv``: per topic-asset pair the contribution and the true
  contribution over the window (decimal return points) and the variance share;
* ``recovery.json``: recovery metrics of the training fit against the truth;
* ``sweep.csv``: OOS R2 over consecutive forecast windows;
* ``summary.json``: config, headline numbers, stage keys and timings;
* with ``--bks``: ``bks_r2.csv`` and ``bks_contrib.csv`` (per-topic split, not
  identified, D52).

Usage (from the repository root)::

    .venv/Scripts/python.exe scripts/run_lab.py                      # LabConfig() defaults
    .venv/Scripts/python.exe scripts/run_lab.py --config lab.yaml --bks --out output/lab/test

The config file is JSON or YAML with the fields of
:class:`~narrative_ipca.exposure_lab.config.LabConfig` (missing fields keep
their defaults). The default output folder is ``output/lab/<config hash>``.

Naming: a topic sensitivity is the expected return response of an asset to a
one-standard-deviation attention shock in a topic, with the other topics'
shocks held fixed. In code, "exposure" means topic sensitivity (the package
``exposure_lab``, the config section ``exposure``). The output files were
``exposure_corr.csv`` and ``exposures.csv`` (column ``w_design``) until
2026-09-30.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from narrative_ipca.config import load_config  # noqa: E402
from narrative_ipca.exposure_lab.config import LabConfig  # noqa: E402
from narrative_ipca.exposure_lab.evaluate import median_finite  # noqa: E402
from narrative_ipca.exposure_lab.session import run_lab  # noqa: E402

logger = logging.getLogger("run_lab")


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.date().isoformat()
    return x


def write_outputs(out: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Write the result files of one :func:`run_lab` call; returns the summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg: LabConfig = out["config"]
    ev, fit, truth, sim = out["evaluation"], out["direct"], out["truth"], out["simulation"]

    ev.corr.to_csv(out_dir / "oos_corr.csv", float_format="%.6g")

    pairs = pd.DataFrame(
        {
            "b_hat": fit.B_hat.stack(),
            "b_true": truth.B_true.reindex(index=fit.B_hat.index, columns=fit.B_hat.columns).stack(),
            "w_set": truth.W.reindex(index=fit.B_hat.index, columns=fit.B_hat.columns).stack(),
            "selected": fit.selected.stack(),
        }
    )
    pairs.index.names = ["topic_id", "asset_id"]
    pairs.to_csv(out_dir / "sensitivities.csv", float_format="%.6g")

    r2 = pd.DataFrame(
        {
            "name": sim.market.assets["name"],
            "asset_class": sim.market.assets["asset_class"],
            "source": sim.market.assets["source"],
            "r2": ev.r2,
            "r2_oracle": ev.r2_oracle,
            "r2_population": truth.r2_true,
            "realized": ev.realized,
            "explained": ev.explained,
            "explained_true": ev.explained_true,
            "residual": ev.residual,
        }
    )
    r2.index.name = "asset_id"
    r2.to_csv(out_dir / "r2.csv", float_format="%.6g")

    contrib = pd.DataFrame(
        {
            "contribution": ev.contrib.stack(),
            "contribution_true": ev.contrib_true.stack(),
            "variance_share": ev.var_share.stack(),
            "variance_share_true": ev.var_share_true.stack(),
        }
    ).add(0.0)  # no "-0" in the file
    contrib.index.names = ["asset_id", "topic_id"]
    contrib.to_csv(out_dir / "contributions.csv", float_format="%.6g")

    (out_dir / "recovery.json").write_text(json.dumps(_jsonable(ev.recovery), indent=2), encoding="utf-8")
    out["sweep"].to_csv(out_dir / "sweep.csv", index=False, float_format="%.6g")

    days = ev.return_days
    summary: dict[str, Any] = {
        "config": cfg.to_dict(),
        "config_hash": cfg.hash(),
        "n_assets": int(len(sim.market.assets)),
        "n_topics": int(len(sim.topics.table)),
        "n_links": int(len(sim.links.table)),
        "forecast_days": int(ev.n_days),
        "forecast_first_day": days[0].date().isoformat() if len(days) else None,
        "forecast_last_day": days[-1].date().isoformat() if len(days) else None,
        "median_r2": median_finite(ev.r2),
        "median_r2_oracle": median_finite(ev.r2_oracle),
        "median_r2_population": median_finite(truth.r2_true),
        "share_positive_r2": float((ev.r2.dropna() > 0).mean()) if ev.r2.notna().any() else float("nan"),
        "recovery": ev.recovery,
        "feasibility_scaled_topics": list(sim.meta.get("feasibility_scaled_topics", [])),
        "clipped_share": sim.meta.get("clipped_share"),
        "failed_assets": list(sim.market.meta.get("failed_assets", [])),
        "keys": out["keys"],
        "timings": out["timings"],
    }
    if "bks" in out:
        res = out["bks"]
        bks_r2 = pd.DataFrame({"r2_bks_weekly": res.r2, "r2_direct_daily": ev.r2})
        bks_r2.index.name = "asset_id"
        bks_r2.to_csv(out_dir / "bks_r2.csv", float_format="%.6g")
        split = res.contrib.copy()
        split.insert(0, "const", res.const_contrib)
        split.index.name = "asset_id"
        split.to_csv(out_dir / "bks_contrib.csv", float_format="%.6g")
        summary["bks"] = {
            "lam": res.lam,
            "K": res.K,
            "selected_topics": list(res.selected_topics),
            "r2_pooled": res.r2_pooled,
            "r2_pooled_shuffled_instruments": res.meta.get("shuffled_r2_pooled"),
            "median_r2": median_finite(res.r2),
            "in_sample_total_r2": res.in_sample_total_r2,
            "n_weeks": int(len(res.periods)),
            "warnings": list(res.meta.get("warnings", [])),
        }
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the topic-sensitivity lab once and write its results.")
    parser.add_argument("--config", type=Path, default=None, help="JSON or YAML LabConfig (default: LabConfig()).")
    parser.add_argument("--bks", action="store_true", help="Also fit and evaluate BKS Sparse IPCA (weekly).")
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default: output/lab/<config hash>).")
    parser.add_argument("--quiet", action="store_true", help="Log warnings only.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    # the configs normalise lists and int-valued numbers themselves (D77)
    cfg = load_config(args.config, cls=LabConfig) if args.config else LabConfig()
    out_dir = args.out if args.out is not None else _ROOT / "output" / "lab" / cfg.hash()

    def progress(done: int, total: int, message: str) -> None:
        print(f"  BKS lambda path {done}/{total}", file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    out = run_lab(cfg, with_bks=bool(args.bks), progress=progress if args.bks else None)
    summary = write_outputs(out, out_dir)
    rec = summary["recovery"]
    lines = [
        f"Topic-sensitivity lab: {summary['n_assets']} assets x {summary['n_topics']} topics, "
        f"{summary['n_links']} links",
        f"  forecast window {summary['forecast_first_day']} to {summary['forecast_last_day']} "
        f"({summary['forecast_days']} return days)",
        f"  median OOS R2: estimator {summary['median_r2']:.3f}, oracle {summary['median_r2_oracle']:.3f}, "
        f"population {summary['median_r2_population']:.3f}; positive for {summary['share_positive_r2']:.0%} of assets",
        f"  recovery: coverage {rec['coverage']:.2f}, sign agreement {rec['sign_agreement']:.2f}, "
        f"MCC {rec['mcc']:.2f}, Spearman {rec['spearman']:.2f}",
    ]
    if "bks" in summary:
        b = summary["bks"]
        lines.append(
            f"  BKS: lambda {b['lam']:.4g}, K {b['K']}, {len(b['selected_topics'])} topics selected, pooled OOS R2 "
            f"{b['r2_pooled']:.3f} over {b['n_weeks']} week(s) (instruments shuffled: "
            f"{b['r2_pooled_shuffled_instruments']:.3f}; per-week factors, not comparable to the direct R2)"
        )
    lines.append(f"  wrote {out_dir} in {time.perf_counter() - t0:.1f}s")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
