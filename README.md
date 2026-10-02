# narrative-ipca

Production implementation of the narrative asset pricing pipeline of Bybee,
Kelly & Su (2023, "Narrative Asset Pricing: Interpretable Systematic Risk
Factors from News Text", BKS), built for a multi-asset universe whose daily
topic attention series are supplied as input (FASTopic upstream). It covers
the full BKS estimation chain, a simulation with known ground truth, and an
evaluation harness that scores what the estimator recovers.

- `DESIGN.md` is the binding specification: the method mapped to the paper's
  equations, the decision register (D1-D52) with every assumption and best
  guess, the module contracts, the simulation design and the harness.
- `reports/simulation/` holds the harness reports (what the estimator
  recovered on simulated data, and how that compares with the truth).

## What it does

```
daily attention levels theta   daily excess returns r
            |                          |
   shocks z = theta - MA5(theta)       |            Step 2  (BKS Section 3.1)
            |                          |
   kernel-weighted covariances cov_{i,t} = Cov(r_i, z)  Step 3  (Eq. 6, xi = 0.99 per month)
            |
   instruments c_{i,t} = [1, cov_{i,t}] paired with r_{i,t+1}   Step 4  (Eq. 7)
            |
   Sparse IPCA: min 1/2 SSR + lambda N_S sum_l sigma_l ||Gamma_l|| + sum_t ||f_t||^2   Step 5  (Eq. 8, ARLS)
            |
   lambda tuned on the MVE Sharpe (in sample, or LOOCV)          Step 6
            |
   A, latent states x, impact vectors I_{z->MVE}                 Step 7  (Eq. 10-12)
   expanding-window OOS factors and MVE portfolio                Step 8  (Section 4.2)
   R2, Sharpe, pricing tests, placebo test                       Step 9
```

Each step is one module (`narrative_ipca/<step>.py`), takes one config
dataclass, and returns one typed container (`narrative_ipca/types.py`).

## Install

```bash
pip install -e ".[dev]"
```

Core dependencies are numpy, pandas and scipy. `numba` (extra `accel`, also
in `dev`) JIT-compiles the group-lasso sweep and is strongly recommended for
universes above ~50 topics; without it the identical pure-numpy path is used
(`NARRATIVE_IPCA_NO_NUMBA=1` forces it).

## Use

Python:

```python
from narrative_ipca.config import PipelineConfig
from narrative_ipca.types import AttentionData, ReturnsData
from narrative_ipca.pipeline import run_pipeline, save_result

attention = AttentionData(levels=theta_daily)          # DataFrame: dates x topics
returns = ReturnsData(returns=excess_daily, asset_meta=meta)  # DataFrame: dates x assets, NaN = not in universe
result = run_pipeline(attention, returns, PipelineConfig())
print(result.fit.selected_topics, result.evaluation.metrics["oos_sharpe"])
save_result(result, "runs/my-run")
```

Command line:

```bash
narrative-ipca run --config configs/default.yaml --attention attention.parquet --returns returns.parquet --out runs/my-run
```

```bash
narrative-ipca simulate --scenario baseline --seed 0 --out runs/sim-baseline
```

```bash
python scripts/run_simulation_study.py --scenarios baseline,no_factor,topic_null,softmax,weak --seeds 3 --out reports/simulation
```

`configs/default.yaml` lists every knob with its BKS default (monthly
periods, 5-day shock window, kernel decay 0.99, K = 3, penalised intercept,
in-sample Sharpe tuning, annual refits in the expanding OOS window).

## Simulation and harness

`simulation.simulate` draws attention and returns from the BKS
data-generating process (Figure 1 of the paper) with known `A`, factors,
states, betas and MVE Sharpe: K = 3 states, 120 topics of which 20 carry
signal and 20 are pure placebos, 500 assets in four asset classes with
class-specific volatility and loadings, an unbalanced panel, 20 years of
daily data. `harness.run_harness` runs the pipeline on each scenario and seed
and compares the output with the truth through rotation-invariant metrics
(selection recall/precision, placebo count, principal angles of the loading
subspace, canonical correlations of factors and states, impact-vector rank
correlation, OOS Sharpe relative to the true MVE). Thresholds live in
`HarnessThresholds`; the report explains, per scenario, what was expected
and what was observed.

Two null scenarios exist for a reason documented in `DESIGN.md` D47: with
priced factors in returns, the kernel covariances of even pure-noise topics
inherit the assets' betas, so the estimator can build priced factor
portfolios from topics that carry no information. `no_factor` is the
chance-level null; `topic_null` shows the mechanism.

## What the simulation study showed (2026-09-06)

Full detail: `reports/simulation/STUDY_NOTES_2026-09-06.md` and the harness
reports under `reports/simulation/full/`. In short:

1. The factor structure is recovered in every signal scenario: implied betas
   at a first canonical correlation of 0.97 with the truth, factors at 0.99,
   93% of the true systematic return explained, and an out-of-sample MVE
   Sharpe of about 70% of what the true factors' MVE realises.
2. With no topic carrying information but priced factors in returns, the
   estimator still recovers betas (0.92) and earns 40% of the true OOS Sharpe,
   because kernel covariances of noise topics inherit the betas (D47). Sharpe
   ratios, selection counts and selection stability therefore cannot certify
   narrative information; the placebo test and pricing errors can.
3. The paper's in-sample argmax rule is not selective on a flat Sharpe
   surface (91 of 120 topics, 15 of 20 placebos); the 2% tolerance rule
   (`TuningConfig.tolerance`) selects 9-11 topics with no placebo in two of
   three seeds (the third lands in the dense region) at the same OOS Sharpe.
4. Loading rows, impact vectors and latent states are not identified by the
   model (D52); the implied betas `c Gamma` are.

## Topic-sensitivity lab and dashboard (2026-09-29)

An interactive lab that shows how much of an asset's return variation news
topics explain out of sample. The spec and decisions are in `DESIGN.md` Part G.

**Terms** (DESIGN.md G.0, D89). **Topic sensitivity** `b_{k,n}` (the matrix
`B`, topics x assets) is the expected return response of asset `n` to a
one-standard-deviation attention shock in topic `k`, with the other topics'
shocks held fixed. It is the coefficient in the regression of the asset's
return on all topics' attention shocks at once,
`r_{n,t+l} = a_n + sum_k b_{k,n} s_{k,t} + e_{n,t+l}`, where `r_{n,t+l}` is
asset `n`'s return on day `t + l`, `s_{k,t}` is topic `k`'s attention shock
on day `t` (attention minus its mean over the previous `w` days, divided by
its standard deviation), `a_n` is an intercept, `e_{n,t+l}` is the part not
explained by topics and `l` is the lead (0 = same day, 1 = next day).

- Units: % return per one-standard-deviation shock, or standardised (the
  return also divided by its standard deviation).
- It is not a position size or dollar exposure: it says how an asset's return
  moves with news attention, not how much of the asset a portfolio holds.
- The lab has three versions: set sensitivities (`W`, the "betas"), true
  sensitivities (`B_true`) and estimated sensitivities (`B_hat`).
- **"Exposure" in code means topic sensitivity.** The code keeps the older
  word: the package `exposure_lab`, `ExposureConfig`, `exposure_heatmap`,
  `B_hat` and the "exposures" in docstrings and variable names all mean topic
  sensitivity. The documents use "topic sensitivity" since 2026-09-30,
  because "exposure" is easily read as a dollar exposure to the asset.

What the lab covers:

- **Assets**: the 55 listed multi-asset spread trades ("A v B" is long leg A,
  short leg B; outrights and "XXX v USD" pairs are long against cash), with
  real daily prices 2015-01-02 to 2025-12-31 for every leg, or 2-500 generic
  assets. Artificial returns replace a leg that fails to download.
- **Topics**: the 20 manual topics of the AM NLP Sentiment Analytics report
  (sector S1-S11, economic A1-A6 and B1-B3), any subset of them, and/or 0-500
  generic topics.
- **Simulation**: each topic's attention is built from the returns of the
  assets it is linked to, plus noise. Links come in three tiers (strong,
  moderate, weak) with one to three set sensitivities ("betas").
- **Estimators**: the research plan's direct sensitivity regression (elastic net
  by default) and BKS Sparse IPCA, both fitted on a training window and scored
  on a separate forecast window of 1-12 weeks.
- **Time windows**: the sidebar sets the training window by its end (cut-off)
  and a length of one month to 10 years counted back; dashboard defaults are
  6 months to 2025-06-30 and a 4-week forecast from 2025-07-01.
- **Views**: explained variation per asset, an asset x topic table of
  correlations or sensitivities (the Correlation table tab) in the layout of
  the desk example, and a per-asset ranking of topic contributions.
- **Compare methods**: elastic net, ridge, OLS, BKS (through its implied
  topic sensitivities) and the oracle scored on the same forecast days with
  sensitivities frozen at the training end, with one method at a time against
  the oracle. BKS enters twice: with the full history before the cut-off (its
  instruments also weigh the days before the training window; the tab states
  how much) and with the training window only, which sees the data the direct
  methods see (DESIGN.md G.15, D88). BKS-implied scores below the direct
  methods; the reasons are in DESIGN.md G.15.1 and the Data and method tab.
- **BKS trace page**: the BKS run of the current settings followed step by
  step (inputs, alignment and scaling, attention shocks, kernel-covariance
  instruments, weekly panel, Sparse IPCA fit and lambda path, forecast weeks,
  implied sensitivities), with an independent reference next to every result
  (a recomputation by the formula, an identity, or the simulation's true
  value), identity and diagnostic checks, and a ladder of variants from the
  true sensitivities down to the BKS-implied ones that shows which step loses
  the signal (DESIGN.md G.16, D90). The BKS tab and the Compare methods tab
  link to it; Run BKS works there too.
- **Real data page**: a placeholder for estimates on real news; it shares the
  sidebar and lists the settings that will apply to real data.

Install and run (from the repository root):

```bash
pip install -e ".[dev,lab]"
```

```bash
python -m streamlit run dashboard/app.py
```

Headless run of one configuration (JSON or YAML `LabConfig`, optional BKS):

```bash
python scripts/run_lab.py --config my_lab.json --bks --out output/lab
```

Regenerate the market data (about 20 seconds online; `--offline` rebuilds from
`data/market/raw/`):

```bash
python scripts/fetch_market_data.py
```

| Path | Content |
|---|---|
| `narrative_ipca/exposure_lab/` | config and types (contracts), reference and market data, links and simulation, direct estimator and evaluation, BKS wrapper, method comparison, BKS trace (`trace.py`), charts, cached session |
| `dashboard/` | the Streamlit app (`app.py`), its pure helpers (`_ui.py`), the BKS trace page's steps (`trace_page.py`) and the Real data page (`real_exposures.py`) |
| `data/reference/` | assets (image order), legs with benchmark index and proxy, the 20 manual topics, the default link map |
| `data/market/` | daily leg levels and returns, asset returns, raw downloads and `manifest.json`; `README.md` there has sources, conventions, QA and open items |
| `tests/test_lab_*.py` | lab tests, including Monte Carlo checks of the simulation truth |

## Layout

| Path | Content |
|---|---|
| `narrative_ipca/config.py` | all configuration dataclasses (one per step), JSON/YAML IO |
| `narrative_ipca/types.py` | typed containers passed between steps |
| `narrative_ipca/data.py`, `shocks.py`, `covariances.py`, `panel.py` | Steps 1-4 |
| `narrative_ipca/grouplasso.py`, `sparse_ipca.py` | Step 5: group lasso (Yang & Zou GMD), ARLS, lambda path, plain IPCA |
| `narrative_ipca/tuning.py`, `wrapup.py`, `oos.py`, `evaluation.py` | Steps 6-9 |
| `narrative_ipca/pipeline.py`, `cli.py` | orchestration, artefacts, CLI |
| `narrative_ipca/simulation.py`, `harness.py` | DGP with ground truth; evaluation harness and report |
| `tests/` | equation-level tests (brute-force references, KKT conditions, invariances) |
| `scripts/` | `run_full_study.py` (parallel study driver), `summarise_study.py`, `rescore_study.py`, `run_simulation_study.py`, `benchmark_solver.py` |
| `reports/simulation/` | study notes, harness reports and tables (`README.md` there has the regeneration recipe) |

## Status

v0.1 (2026-09-06). Research code that will be refactored; every deliberate
choice is in `DESIGN.md` so that a change starts from a stated reason.
Known limitations are listed in `DESIGN.md` Part F.
