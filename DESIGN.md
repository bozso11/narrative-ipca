# narrative-ipca — design and decision register

Production implementation of Bybee, Kelly & Su (2023), *Narrative Asset Pricing:
Interpretable Systematic Risk Factors from News Text* (BKS), with the estimator
theory of Kelly, Korsaye, Pruitt & Su (2026), *Instrumented Principal Component
Analysis* (KKPS). Status: v0.1, 2026-09-06 (updated after the verification round of the same day; see D46-D49). Written for a multi-asset universe
whose topic attention time series come from FASTopic and are supplied as input.

The document has seven parts:

- Part A — the methodology, step by step, mapped to modules and equations.
- Part B — the decision register (D1–D52): every assumption or best guess, with
  the reason and where to change it. The lab's decisions D53–D72 and D74–D90
  are in Part G (D73 is a Part B entry).
- Part C — module contracts (function signatures the code implements).
- Part D — the simulation data-generating process with known ground truth.
- Part E — the evaluation harness and what the simulated data should show.
- Part F — open points and known limitations.
- Part G — the topic-sensitivity lab and dashboard (2026-09-29; code package
  `exposure_lab`): simulated topics built from real multi-asset prices, the
  direct sensitivity regression and BKS compared out of sample, and the
  Streamlit dashboard. G.0 defines **topic sensitivity**, the quantity the
  code calls "exposure". G.16 describes the BKS trace page, which follows a
  lab BKS run step by step against references (D90).

Symbols are defined at first use. Equation numbers refer to the BKS working
paper (3 May 2023 version); "KKPS Alg. 1" is the ALS pseudocode in the IPCA
paper's Appendix J.

---

## Part A — Methodology to module map

The pipeline is the three-step estimation procedure of BKS Section 2 wrapped
with input handling, tuning, out-of-sample evaluation and interpretation.

| Step | What it does | BKS reference | Module |
|---|---|---|---|
| 0 | Configuration | — | `config.py` |
| 1 | Input validation and calendar alignment | Section 3.1 (timing convention) | `data.py` |
| 2 | Attention shocks `z_tau` | Section 3.1, App. C.5 | `shocks.py` |
| 3 | Kernel-weighted covariances `cov_{i,t}` | Eq. 6, App. B.1 | `covariances.py` |
| 4 | Estimation panel `c_{i,t-1} -> r_{i,t}`, `sigma^c_l` | Eq. 7, App. B.1 | `panel.py` |
| 5 | Sparse IPCA by alternating regularised least squares | Eq. 8, Eq. 16, App. B.2 | `grouplasso.py`, `sparse_ipca.py` |
| 6 | `lambda` (and `K`) tuning | Section 2 (footnote 8), App. C.3 | `tuning.py` |
| 7 | Wrap-up: `A`, states `x_tau`, impact vectors | Section 2 step 3, Section 6.1 | `wrapup.py` |
| 8 | Expanding-window out-of-sample factors and MVE | Section 4.2 | `oos.py` |
| 9 | Evaluation: R2, Sharpe, pricing tests, placebo test | Sections 3.2, 4.1, App. C.2 | `evaluation.py` |
| 10 | Orchestration, artefacts, CLI | — | `pipeline.py`, `cli.py` |
| — | Simulation with ground truth; harness | App. C.2 (placebo idea), KKPS App. G.1 | `simulation.py`, `harness.py` |

### A.1 The model being estimated

- **State variables** `x_tau` (K x 1): the ICAPM state variables on day `tau`.
- **Narrative shocks** `z_tau` (L x 1): innovations to attention on L topics.
  Eq. 1: `z_tau = A x_tau + eta_tau`, with `A` (L x K) row-sparse.
- **Factors** `f_t` (K x 1): the projection of `x` on the space of excess returns
  (state-mimicking portfolios); `x = f + nu` with `nu` orthogonal to returns.
- **Returns** Eq. 4: `r_{i,t+1} = beta_{i,t} f_{t+1} + eps_{i,t+1}`.
- **Instruments** Eq. 5: `cov_{i,t} := Cov_t(r_{i,t+1}, z_{t+1}) = beta_{i,t} Sigma_ff A'`,
  inverted as `beta_{i,t} = cov_{i,t} Gamma_tilde` with
  `Gamma_tilde = A (A'A)^-1 Sigma_ff^-1` (L x K).
- **Estimation model** Eq. 7: `r_{i,t+1} = c_{i,t} Gamma f_{t+1} + e_{i,t+1}`,
  `c_{i,t} = [1, cov_hat_{i,t}]`, `Gamma = [Gamma_0; Gamma_tilde]` ((L+1) x K).

### A.2 The estimator (Eq. 8)

```
min_{Gamma,{f_t}}  1/2 sum_{i,t in S} (r_{i,t} - c_{i,t-1} Gamma f_t)^2
                 + lambda N_S sum_{l=0..L} sigma^c_l ||Gamma_l||_2
                 + sum_{t in S} ||f_t||_2^2
```

In words: a least-squares fit of period returns on instrumented loadings times
factors, plus a group lasso over the rows of `Gamma` (a narrative is in or out
for all K factors at once), plus a ridge on the factors that pins down the
scale (without it `Gamma` could shrink to zero and `f` grow without bound,
App. B.1). `N_S` is the number of asset-period observations in the training
sample `S`, and `sigma^c_l` is the panel standard deviation of instrument `l`
over `S` (`sigma^c_0 = 1`), which puts the penalty of every narrative on the
same scale without rescaling the instruments themselves.

The solver (App. B.2) alternates:

1. **f-step**, closed form per period (Eq. 16):
   `f_t = (Gamma' C_{t-1}' C_{t-1} Gamma + 2 I_K)^-1 Gamma' C_{t-1}' r_t`.
2. **Gamma-step**, a group lasso on `vect(Gamma)` with regressors
   `c_{i,t-1} (x) f_t'`, solved by the groupwise-majorisation-descent algorithm
   of Yang & Zou (2015) on the Gram form
   `G = sum_t (C_{t-1}' C_{t-1}) (x) (f_t f_t')`, `b = sum_t (C_{t-1}' r_t) (x) f_t`.

Both steps are exact block minimisations, so the objective is non-increasing
by construction; that is the convergence guarantee (KKPS App. D, Prop. 4:
convergence to a stationary point, no global-optimality claim).

### A.3 Tuning, wrap-up, out-of-sample

- `lambda` is chosen to maximise the in-sample annualised MVE Sharpe ratio
  `sqrt(12 mu_f' Sigma_ff^-1 mu_f)` along a regularisation path (BKS Section 2),
  with LOOCV (App. C.3) as an alternative.
- Wrap-up: `Sigma_ff` = sample covariance of `f_t`; `A = Gamma_tilde (Gamma_tilde' Gamma_tilde)^-1 Sigma_ff^-1`;
  `x_tau = (A'A)^-1 A' z_tau`; `b_MVE = mu_f' Sigma_ff^-1`; impact vectors
  `I_{z->x} = A (A'A)^-1`, `I_{z->MVE} = I_{z->x} b_MVE'` (Eq. 10–11), term level
  `I_{w->MVE} = Phi (Phi'Phi)^-1 I_{z->MVE}` (Eq. 12) when a topic-term matrix exists.
- Out of sample (Section 4.2): with `Gamma`, `mu_f`, `Sigma_ff` frozen from an
  expanding training window, `f^OOS_{t+1} = (sum_i beta_i beta_i' + 2 I_K)^-1 sum_i beta_i r_{i,t+1}`
  with `beta_i = c_{i,t} Gamma`, and `f^{MVE,OOS} = b_MVE f^OOS`.

---

## Part B — Decision register

Each entry: the decision, why, and the config field or code location to change it.

### Inputs and alignment (`data.py`)

- **D1 Attention input is daily topic levels.** `AttentionData.levels` is a
  dates x topics DataFrame of attention levels `theta_tau`. Aggregation from
  FASTopic document-topic distributions to a daily level is upstream; a helper
  `aggregate_documents()` implements the BKS term-count-weighted mean
  (`theta_tau = sum_m N_m theta_m / sum_m N_m`) for users who hold document-level
  output. Rows are not required to sum to one.
- **D2 Return input is daily excess returns per asset,** `NaN` where the asset
  is not in the universe. `return_kind="total"` plus a risk-free series is
  supported. Multi-asset: any instrument with a daily excess return can be a
  column; an `asset_meta` table (e.g. `asset_class`) is carried to reports.
- **D3 Calendar = the return calendar.** Attention is re-indexed to the trading
  days of the returns panel. Attention on non-trading days is dropped by
  default (`non_trading_day_policy="drop"`); `fold_mean`/`fold_sum` fold it into
  the next trading day. Reason: BKS work on the trading-day grid; how weekend
  news should be folded is an empirical choice for the pilot.
- **D4 Timing convention.** `theta_tau` is assumed synchronised with `r_tau`
  (BKS: the edition published the morning of `tau+1` reflects day `tau`).
  `attention_lag_days` shifts attention forward if the upstream stamping
  differs. No look-ahead is introduced anywhere downstream because instruments
  at `t` only use days up to `skip_days` before the end of period `t`.
- **D5 Estimation period is the calendar month** (`period="M"`), as in BKS.
  Any pandas offset alias works; the kernel decays per period.
- **D6 Period return = sum of daily excess returns** (`return_aggregation="sum"`).
  Reason: BKS treat `r`, `z`, `x` as innovations that "can be accumulated from
  daily to monthly frequencies" (Section 1) and the factors are linear
  portfolios. `compound` is available; it is not exactly consistent with the
  linear factor structure but matches CRSP monthly returns more closely.
- **D7 Asset weighting.** The pooled SSR weights every asset-period equally
  (`asset_weighting="none"`, BKS). In a universe mixing equities, credit and
  rates, high-volatility assets dominate the fit. The model is
  scale-equivariant per asset (scaling `r_i` scales `cov_i` identically, so
  `Gamma` and `f` are unchanged for the narrative part of the loading; the
  intercept row `Gamma_0` is not rescaled, so the equivariance is exact only
  when `Gamma_0 = 0`), hence `inverse_vol` scaling is essentially a
  reweighting of assets in the objective. It is offered as an option; the
  harness accepts a pipeline config, so its effect can be measured by running
  the harness twice (not a built-in comparison, TBD). Factors under
  `inverse_vol` are portfolios of vol-scaled positions.
- **D8 Periods with fewer than `min_assets_per_period` observed assets are
  dropped** from the panel (their `f_t` would be noise).

### Shocks (`shocks.py`)

- **D9 `z_tau = theta_tau - (1/w) sum_{j=1..w} theta_{tau-j}`, `w = 5`,** trailing
  and strictly prior, on the trading-day grid (rows, not calendar days). `w=1`
  is the daily difference; `3` and `20` are the BKS robustness variants.
- **D10 No standardisation by default.** Eq. 8 already rescales the penalty by
  `sigma^c_l`; standardising `z` is a display convenience and not OOS-safe.
- **D10a Simplex inputs.** If daily attention rows sum to one (LDA and
  FASTopic both produce this at the document level), shocks sum to zero
  across topics and the covariance instruments are linearly dependent
  (`sum_l cov_{i,l} = 0`). The estimator tolerates this: the Gamma-step Gram
  is singular only at `lambda = 0`, which is handled by plain IPCA with a
  pseudo-inverse (D18). Nothing is dropped.

### Covariances (`covariances.py`)

- **D11 Kernel decays per period, not per day** (App. B.1: `kappa(tau;t) =
  xi^(t - t_tau)`, `t_tau` the period of day `tau`). `xi = 0.99` per month is a
  69-month half-life. Every day of period `t` (up to the window end) has raw
  weight 1.
- **D12 The window for `cov_{i,t}` ends `skip_days = 1` trading days before the
  last day of period `t`** (footnote 9: "up to the second last day"), so the
  instrument is known before the period-`t+1` return accrues.
- **D13 Weighted covariance, not cross-moment:** Eq. 6 subtracts the product of
  the two weighted means, using weights renormalised over the days on which
  the asset actually has a return and the shock is available.
- **D14 Missing days.** Available-case per asset: weights renormalise over
  observed asset-days; an asset-period needs `min_days` observed days in the
  window, otherwise `cov_{i,t}` is `NaN` and the asset is out of that period's
  cross-section.
- **D15 Full history by default** (`lookback_periods=None`, BKS). Truncation is
  available for speed and to shorten memory; with `xi=0.99` a 120-period
  truncation drops weights below 0.30.
- **D16 Implementation via per-period sufficient statistics.** Because weights
  are constant within a period, `cov_{i,t}` is an exponentially weighted
  recursion over per-period sums of `r z'`, `r`, `z` and day counts. Cost is
  `O(T N L)` after one pass over the daily data; the current period's partial
  sum (excluding the last `skip_days` days) is handled separately.

### Panel (`panel.py`)

- **D17 Row `(i, t)` pairs `c_{i,t-1}` with `r_{i,t}`;** both must be finite. The
  first `burn_in_periods` instrument periods are discarded.
- **D26 `sigma^c_l` is computed on the training sample only** (population std,
  `ddof=0`, column 0 fixed at 1) and is recomputed for every sub-sample
  (`IPCAPanel.subset_periods`). Reason: App. B.1 defines it on `S`; reusing a
  full-sample value in an expanding-window scheme would leak.

### Sparse IPCA (`grouplasso.py`, `sparse_ipca.py`)

- **D18 `lambda = 0` is plain IPCA, not Eq. 8.** At `lambda = 0` Eq. 8 has no
  scale identification (the ridge pushes `f` to zero and `Gamma` to infinity).
  The unregularised case is therefore estimated by KKPS Alg. 1 (ALS with the
  `Theta_Y` normalisation `Gamma'Gamma = I_K`, factor second moment diagonal
  descending), and reported through the same `SparseIPCAResult` with
  `selected` all True. The regularisation path uses `lambda > 0` only.
- **D19 The intercept row is penalised** (`penalize_intercept=True`) with
  `sigma^c_0 = 1`, following the sum `l = 0..L` in Eq. 8.
- **D20 Initialisation:** top-K left singular vectors of the managed-portfolio
  matrix with columns `C_{t-1}' r_t / N_t` (KKPS Alg. 1 initialisation), a tiny
  seeded jitter, then `n_warmup = 3` unpenalised ARLS sweeps to put `Gamma`
  on the right scale before the penalty is applied. Reason: starting the
  penalised problem from a badly scaled `Gamma` zeros every row at once.
- **D21 Gamma-step solver:** Yang & Zou (2015) groupwise majorisation descent
  on the Gram form, warm-started from the previous sweep. Block curvature is
  `lambda_max(G_gg)` per group. Every update is a descent step on the exact
  objective, so `obj_path` is non-increasing; tests assert this.
- **D22 Lambda grid:** `n_lambdas` log-spaced points between `ratio * lam_max`
  and `lam_max`, where `lam_max` is the smallest penalty at which every
  narrative row is zero given the warm-up factors (from the KKT condition
  `||b_l - (G x*)_l|| <= lambda N_S sigma_l` with only the constant row active).
  With the intercept penalised (D19) the constant row itself is
  soft-thresholded, so `x*` is the exact minimiser of the one-group problem
  and `lam_max` is found by bracketing (verification fix, 2026-09-06).
  `lam_max` is conditional on the warm-up factors, so it moves with
  `n_warmup`; it positions the grid, it does not change any fit at a given
  `lambda`. The path is traced in ascending `lambda` with warm starts (dense
  to sparse, the numerically stable direction). Points near `lam_max` can be
  spurious non-trivial stationary points (objective above the all-zero
  solution); the tuner never picks them because their Sharpe is low. At the
  other end, the study grid extends to `1e-4 lam_max` so that the in-sample
  argmax does not sit on the dense boundary (the criterion is flat down to
  about `1e-3 lam_max`); the densest points may hit `max_iter = 500` outer
  sweeps and are then reported with `converged = False` (their objective is
  still non-increasing and their fit is usable).
- **D23 Convergence:** relative change of the Eq. 8 objective below `tol`
  (default 1e-8) or `max_iter` sweeps. Both the value and the path are
  returned. The inner group-lasso budget (`inner_max_iter = 200` sweeps,
  `inner_tol = 1e-10`) is not reached at the dense end of the path on the
  full-size panel (the Gram there has condition number about `5e4` even in
  standardised coordinates, D50). This is immaterial: alternating descent
  does not need exact block minimisation, warm starts accumulate sweeps, and
  raising the budget to 5,000 sweeps left objective, selection, `R2` and
  Sharpe unchanged to the sixth digit while costing 6x the time (experiment
  of 2026-09-06 at `1e-3`, `1e-2` and `1e-1` of `lam_max`). The outer
  tolerance sets the accuracy; `meta["inner_all_converged"]` records the
  inner status.
- **D24 Identification and canonical form.** Eq. 8 is invariant to orthogonal
  rotations `Gamma -> Gamma Q`, `f -> Q' f` (row norms and `||f||` are
  preserved) but not to general invertible rotations or rescaling (the penalty
  balance pins the scale). `canonicalize()` applies only orthogonal rotations:
  `Sigma_ff` diagonal with descending entries, signs chosen so that each
  factor's mean is non-negative (fallback: first non-zero entry of each
  `Gamma` column positive, KKPS App. G.2 [3'']). It changes labels only; all
  reported statistics are invariant to it.
- **D25 Factor moments** `mu_f`, `Sigma_ff` use `ddof = 1` over populated
  periods only.

### Tuning (`tuning.py`)

- **D27 Criterion = in-sample annualised MVE Sharpe** (BKS main text, footnote
  8). Ties break toward the sparser solution. `TuningConfig.tolerance`
  (default 0 = the BKS exact argmax) widens the tie band to every point
  within a relative distance of the maximum, so the sparsest point within,
  say, 2% of the best Sharpe wins. Reason (full-size study, 2026-09-06): on
  the simulated panel the in-sample Sharpe surface is flat (1.053-1.057) from
  91 down to 10 selected narratives, so the exact argmax admits dozens of
  noise narratives that do not move the criterion. The study reports the
  argmax, the tolerance rule and LOOCV side by side.
- **D28 LOOCV (App. C.3)** is implemented as: for each left-out period `t`,
  fit on `S \ {t}`, form `f^MVE_t` from the left-out cross-section with the
  OOS formula, stitch, compute the Sharpe ratio. It costs `T` fits per lambda;
  `loocv_max_folds` subsamples the left-out periods.
- **D29 `K` is fixed at 3 by default** (BKS "to be conservative"); a joint
  `(lambda, K)` search over `K_grid` uses the same criterion.

### Out-of-sample (`oos.py`)

- **D30 Expanding window,** first OOS period from `first_oos_period` or the
  last `oos_fraction` of periods; **refit every 12 periods**; `Gamma`, `mu_f`,
  `Sigma_ff`, `lambda` frozen in between (BKS retrain each December).
- **D31 `lambda` is retuned at every refit** (`retune_lambda=True`); the
  `sigma^c_l` of the training window is used for the frozen `Gamma`.
- **D32 OOS factor extraction** uses the ridge `2 I_K` exactly as in Eq. 16
  (the ridge is inherited from the objective, not a free parameter).
- **D33 The OOS MVE weights `b_MVE` come from the training window** moments
  of the in-sample factors, as in BKS.

### Evaluation (`evaluation.py`)

- **D34 Sharpe ratios are annualised with `annualization` periods per year**
  (12 for monthly), both for in-sample MVE and for realised OOS series
  (`mean / std * sqrt(12)`, `ddof = 1`).
- **D35 Pricing tests** regress test-asset excess returns on the factors with
  an intercept; alphas are reported in the return units of the inputs
  (decimal per period); GRS uses the finite-sample F form and needs
  `T > N_assets + K + 1`, otherwise `None`.
- **D36 Placebo test (App. C.2):** `placebo_n` i.i.d. normal narratives with
  variances matched to randomly chosen real narratives are appended at the
  *shock* level, the whole pipeline from Eq. 6 onward is re-run, and the
  report states how many placebos survive at the tuned `lambda` and the
  per-instrument `lambda_max` (largest `lambda` at which the row is still
  selected along the path).

### Simulation and harness

- **D37 The DGP follows BKS Figure 1** exactly, with the attention-level
  mapping and asset heterogeneity added (Part D).
- **D38 Ground truth is compared only through rotation-invariant quantities**
  (Part E), because `Gamma`, `f`, `x` are identified up to rotation.
- **D39 Sample sizes** for the headline scenario: 500 assets in four asset
  classes, 120 topics (20 relevant, 20 placebo, 80 irrelevant persistent),
  K = 3, 20 years of 252 trading days, monthly periods, three seeds. This is
  the smallest size at which selection metrics have a small Monte Carlo error
  while the full harness runs in minutes.
- **D40 Thresholds are best guesses** (`HarnessThresholds`) and are
  documented per metric in Part E; they are pass/fail signals for regression
  testing, not statistical tests.

### Engineering

- **D41 Dependencies:** numpy, pandas, scipy for the estimator; `numba` as an
  optional accelerator of the Gamma-step (D46; the estimator runs without it,
  about 30x slower at the dense end of the path); pyarrow and PyYAML optional
  for IO; statsmodels optional (not required: the pricing tests are
  implemented directly).
- **D42 Every stage is a pure function of its inputs and config;** no global
  state; results are dataclasses with `to_frame` helpers; artefacts are
  written by `pipeline.save_result` as parquet/CSV/JSON.
- **D43 Numerics:** float64 in the estimator; pseudo-inverses with `rcond`
  wherever a matrix can be singular by construction (`Sigma_ff` when fewer
  than K narratives are selected, `A'A`); every solver returns convergence
  flags instead of raising. `EvaluationConfig.rcond = 1e-6` (relative): a
  factor direction whose variance is below `1e-6` of the largest is treated
  as dead. Reason (verification finding, 2026-09-06): with the `1e-12` cut a
  numerically dead direction (tiny mean, tiny variance) contributed a large
  spurious `mu^2/sigma^2` to the in-sample Sharpe criterion and to the OOS
  MVE weights exactly at the sparse end of the path. The tuner and the OOS
  loop also flag every fit whose `Sigma_ff` is truncated.
- **D44 Logging** through the standard `logging` module (`narrative_ipca.*`
  loggers); progress callbacks for long loops.
- **D45 Deviation from BKS that is deliberate:** FASTopic instead of LDA
  (attention rows are optimal-transport allocations, still on the simplex);
  multi-asset universe with optional inverse-vol weighting; term-level
  interpretation optional (needs `phi`).
- **D73 Timestamps in `.npz` artefacts are `datetime64[ns]`** (`periods` and
  `window_end` in `covariances.npz`, `periods` in `truth.npz`), not int64
  epoch counts. Reason (2026-09-29): pandas 3 holds datetimes in
  microseconds, so the old `asi8` integers read back as 1970 timestamps; the
  unit now travels with the dtype and `pd.DatetimeIndex(npz["periods"])`
  restores the instants under any pandas version.

### Added after the verification round (2026-09-06)

- **D46 Solver acceleration.** The group-lasso sweep is JIT-compiled with
  `numba` when it is installed (optional extra `accel`); the pure-numpy
  implementation is the reference and is used otherwise or when
  `NARRATIVE_IPCA_NO_NUMBA=1`. Both paths run the identical algorithm and
  agree to rounding on well-posed panels (up to the orthogonal rotation of
  D24 on degenerate ones, where two backends can stop on rotation-equivalent
  stationary points). Reason: on the full-size study panel one fit at the
  dense end of the path took 57 s in pure Python loops; with the kernel it
  takes 1.8 s with identical sweep counts, selection and objective.
- **D47 Two null scenarios, not one.** The original `null` (topics carry no
  information, returns keep their priced factors) is *not* a chance-level
  null for this estimator: the kernel covariance of any noise topic `l` with
  asset `i` is `beta_i' G_{t,l} + noise`, where `G_{t,l} = sum_tau w f_tau z_{l,tau}`
  is a common `K`-vector that is non-zero at `O(1/sqrt(n_eff))` and persists
  over `t` because the kernel half-life is 69 months. With `L` such
  instruments the cross-section spans `beta`, Sparse IPCA recovers `beta`
  from pure-noise topics, the factor portfolios load on the true factors and
  earn the premium. Selection above chance and a positive OOS Sharpe are the
  correct behaviour of the estimator there. The harness therefore runs
  `no_factor` (no common factor structure in returns) as the chance-level
  null with pass/fail checks, and `topic_null` (alias `null`) as a
  report-only scenario whose paragraph documents the mechanism. Because
  `G_{t,l}` persists with the kernel, the spurious selection is also *stable*
  across annual refits: stability of the selected set is a diagnostic of
  estimation noise (low only without any factor structure), not evidence
  that narratives carry information. Consequence for the research plan: an
  OOS Sharpe ratio, even a placebo-free one, cannot certify that narratives
  carry information; the relative placebo test (BKS App. C.2: real
  narratives must beat variance-matched placebos), pricing errors and the
  `topic_null` comparison are the signal-quality evidence (Part F point 2).
- **D48 Calendar consistency in the simulation.** The simulated calendar is a
  business-day range (about 21.7 trading days per calendar month), so the
  daily moments are the period moments divided by the *realised* mean days
  per period `d_bar = n_days / T`, and the daily idiosyncratic standard
  deviation is `idio_vol / sqrt(12 d_bar)`, so that the period-level
  population moments are exact.
- **D49 Contract notes accepted from the implementers.** `build_covariance_panel`
  takes an extra `dtype` keyword (forwarded from `DataConfig.dtype`);
  `project_observable` takes an optional `period` keyword to match
  month-end-stamped observables to last-trading-day-stamped factors and fits
  an intercept (BKS write the projection without one; ICAPM implies zero);
  placebo variances are matched to randomly chosen real topics (with
  replacement) rather than one-to-one; `lambda_path` reports the MVE Sharpe
  with 12 periods per year and the tuner recomputes it with the configured
  annualisation; LOOCV folds are warm-started from the full-sample fit (a
  second-order start-point effect, documented in `tuning.py`); `first_oos_period`
  matches on the period's last trading day (give the first calendar day of
  the intended period); `attention_lag_days = k` means day `tau` carries
  `theta_{tau-k}`; with `retune_lambda = False` and `lam = None` the tuning
  runs once at the first refit.
- **D50 Standardised coordinates inside the solver.** The Gamma-step is
  solved in standardised instrument coordinates: with `sigma = sigma^c`
  (`sigma_0 = 1`), `X_tilde = X diag(1/sigma)` and `Gamma_tilde = diag(sigma) Gamma`,
  so that `X Gamma = X_tilde Gamma_tilde` and the Eq. 8 penalty becomes a
  uniform group lasso `lambda N_S sum_l ||Gamma_tilde_l||`; the f-step is
  invariant. This is an exact reparametrisation of the same objective (BKS's
  point that the regressors are not standardised is respected: `Gamma` is
  reported in original units). Reason (verification finding, 2026-09-06):
  covariance instruments have scales `1e-7..1e-5` against the constant
  column, the Gram had condition number `1e15`, the inner solver could not
  converge within its budget, and the fit was not invariant to the units of
  the attention series. The SVD initialisation now happens in the
  standardised coordinates (scale-free start); the estimator is unchanged.
  Because the start moved, `lam_max` (D22) moved with it (+13% on the
  full-size panel), which only repositions the grid; cold-start fits at a
  fixed `lambda` can land on a different stationary point of the same
  objective than before (KKPS Prop. 4 guarantees a stationary point, not
  the global optimum), which matters for `run_oos` with
  `retune_lambda = False` (cold refits) and not for the warm-started path.
- **D51 Tolerance rule for lambda** (see D27): `TuningConfig.tolerance`,
  default 0 (BKS exact argmax); the study reports the argmax, the 2%
  tolerance rule and LOOCV side by side.
- **D52 What the model identifies, and which harness checks follow from it**
  (full study, 2026-09-06). Eq. 5 gives `cov_{i,t} = beta_{i,t} Sigma_ff A'`,
  so the population instrument vector has rank `K`, and any `Gamma_tilde`
  with `Sigma_ff A' Gamma_tilde = I_K` reproduces every beta:
  `Gamma_tilde = A (A'A)^-1 Sigma_ff^-1 + N` for any `N` with `A'N = 0`, an
  `(L - K)`-dimensional family. In population `K` relevant instruments
  suffice to invert, and the group lasso prefers a sparse representative;
  in finite samples more narratives are selected because they average out
  the noise in the sample covariances. Consequences:
  1. Row-level loadings (`gamma_subspace_cos`), the impact vector built
     from `A_hat` (`impact_spearman`) and the latent states built from
     `A_hat` (`state_canonical_corr`) are not identified targets; the
     harness reports them but does not pass/fail them. The study measured
     mean principal-angle cosines of about 0.6 and impact-vector rank
     correlations of 0.45-0.65 while the implied betas, factors and
     systematic returns were recovered at 0.97, 0.99 and 0.94: the model
     pins down `c Gamma`, not `Gamma`. For the research plan this means that
     BKS's narrative interpretation (which topics, by how much) is a property
     of the sparse representative the lasso happens to pick, not of the
     data-generating `A`; per-topic attribution needs a separately identified
     model (the plan's topic-sensitivity layer), as the vault's anchor note already
     argues on rotation grounds.
  2. Selection recall is not a requirement of the model (a perfect sparse
     fit may use only the strong topics); it is kept as a soft check. Precision
     and the placebo count are the meaningful selection checks, and they
     judge the tuning rule: the BKS exact argmax selected 91 of 120 topics on
     average (precision 0.19, 15 of 20 placebos), the 2% tolerance rule 34
     (precision 0.74, 5 placebos), with identical factor recovery and OOS
     Sharpe.
  3. Under `no_factor` a chance-level selection contains placebos at rate
     `n_placebo / L`, so the placebo check is not applied there.

---

## Part C — Module contracts

Type names refer to `narrative_ipca.types`; config names to
`narrative_ipca.config`. All functions are pure.

### `data.py`

```python
def align_inputs(attention: AttentionData, returns: ReturnsData, cfg: DataConfig) -> AlignedData
    # 1. validate; 2. build daily excess returns (subtract risk_free if return_kind == "total");
    # 3. reindex attention to the return calendar per non_trading_day_policy;
    # 4. apply attention_lag_days (shift forward); 5. optional inverse-vol scaling (trailing,
    #    ex ante: scale on day tau uses days < tau); 6. drop leading/trailing days with no data.
def aggregate_documents(theta_docs: pd.DataFrame, dates: pd.Series, weights: pd.Series | None) -> pd.DataFrame
    # BKS daily aggregation theta_tau = sum_m N_m theta_m / sum_m N_m (weights default to 1).
def period_end_index(calendar: pd.DatetimeIndex, period: str) -> tuple[np.ndarray, pd.DatetimeIndex]
    # (period id per day, last trading day of each period).
def period_returns(returns: pd.DataFrame, period: str, how: str) -> pd.DataFrame
    # sum or compound daily excess returns per period; NaN if no day observed;
    # index = period last trading day.
```

### `shocks.py`

```python
def attention_shocks(attention: pd.DataFrame, cfg: ShockConfig) -> ShockPanel
def shock_diagnostics(shocks: ShockPanel) -> pd.DataFrame   # per topic: std, ac1, share of zero days
def append_placebos(shocks: ShockPanel, n: int, seed: int) -> tuple[ShockPanel, np.ndarray]
    # i.i.d. normal columns "placebo_k" with variance matched to a random real topic (App. C.2);
    # returns (new panel, boolean mask of placebo columns)
```

### `covariances.py`

```python
def kernel_weights(period_id: np.ndarray, t: int, xi: float, lookback: int | None) -> np.ndarray
    # reference implementation for tests: raw weight xi^(t - period_id) for days in periods <= t, else 0
def build_covariance_panel(shocks: ShockPanel, returns: pd.DataFrame, cfg: CovarianceConfig, period: str) -> CovariancePanel
    # Eq. 6 with the per-period recursion of D16; the window of period t excludes its last skip_days days.
def brute_force_covariance(r: pd.Series, z: pd.DataFrame, weights: np.ndarray) -> np.ndarray
    # O(days x L) reference used by the tests to check build_covariance_panel to 1e-10.
```

### `panel.py`

```python
def build_panel(cov: CovariancePanel, returns: pd.DataFrame, cfg: DataConfig, cov_cfg: CovarianceConfig) -> IPCAPanel
    # rows (i, t): X = [1, cov[t-1, i, :]], y = period return of asset i in period t; drop non-finite;
    # drop the first burn_in_periods instrument periods and periods with < min_assets_per_period rows;
    # sigma_c = compute_sigma_c(X).
def split_periods(panel: IPCAPanel, first_oos: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]
    # boolean masks (train, test) over panel.periods
def panel_summary(panel: IPCAPanel) -> pd.DataFrame  # per period: n assets, mean/std of y
```

### `grouplasso.py`

```python
def group_soft_threshold(v: np.ndarray, t: float) -> np.ndarray
def group_lasso_gram(G, b, group_size, penalties, x0=None, max_iter=200, tol=1e-10) -> tuple[np.ndarray, dict]
    # minimise 0.5 x'Gx - b'x + sum_g pen_g ||x_g||; contiguous groups; GMD (Yang & Zou 2015);
    # info: n_iter, converged, max_change, objective. Groups with zero curvature are set to zero.
def group_lasso_objective(G, b, x, group_size, penalties) -> float
def kkt_violation(G, b, x, group_size, penalties) -> float
    # max over groups of the KKT residual; ~0 at the optimum (tests use this, not a reference solver)
```

### `sparse_ipca.py`

```python
def fit_sparse_ipca(panel: IPCAPanel, cfg: EstimationConfig, lam: float | None = None,
                    K: int | None = None, Gamma_init: np.ndarray | None = None,
                    penalize_intercept: bool | None = None) -> SparseIPCAResult
    # lam None -> cfg.lam (must not be None); lam == 0 -> fit_ipca; else ARLS per Part A.2.
def fit_ipca(panel: IPCAPanel, K: int, cfg: EstimationConfig) -> SparseIPCAResult
    # KKPS Alg. 1 (ALS) with Theta_Y normalisation; sets lam=0, selected all True.
def f_step(S: np.ndarray, V: np.ndarray, Gamma: np.ndarray, ridge: float = 2.0) -> np.ndarray   # Eq. 16 for all t
def gram_from_moments(S, V, F) -> tuple[np.ndarray, np.ndarray]                                   # G, b of Part A.2
def objective_value(panel, Gamma, F, lam, sigma_c, penalize_intercept=True) -> float             # Eq. 8
def lambda_max(panel: IPCAPanel, cfg: EstimationConfig, K: int | None = None) -> float             # D22
def lambda_grid(panel, cfg, K=None) -> np.ndarray
def lambda_path(panel, cfg, lams=None, K=None, progress=None) -> tuple[list[LambdaPathPoint], list[SparseIPCAResult]]
    # ascending lambda, warm-started; returns points and the fits (fits needed for tuning/LOOCV)
def canonicalize(res: SparseIPCAResult) -> SparseIPCAResult                                        # D24
def fitted_values(panel, Gamma, F) -> np.ndarray
def betas(X: np.ndarray, Gamma: np.ndarray) -> np.ndarray                                          # c Gamma
```

### `tuning.py`

```python
def tune(panel: IPCAPanel, est_cfg: EstimationConfig, tune_cfg: TuningConfig, eval_cfg: EvaluationConfig,
         progress=None) -> TuningResult
def is_sharpe_criterion(res: SparseIPCAResult, annualization: float) -> float
def loocv_sharpe(panel, est_cfg, lam, K, annualization, max_folds=None, seed=0) -> float
```

### `wrapup.py`

```python
def recover_A(Gamma_tilde: np.ndarray, Sigma_ff: np.ndarray, rcond: float) -> tuple[np.ndarray, bool]
    # A = Gt (Gt'Gt)^-1 Sigma^-1 ; flag rank deficiency (fewer than K non-zero rows)
def state_variables(A: np.ndarray, shocks: ShockPanel, rcond: float) -> pd.DataFrame  # x_tau = (A'A)^-1 A' z_tau
def impact_vectors(A, b_mve, rcond) -> tuple[np.ndarray, np.ndarray]                   # I_{z->x}, I_{z->MVE}
def project_observable(F: pd.DataFrame, target: pd.Series) -> tuple[np.ndarray, float] # b_obs, R2 (Section 6.1)
def term_impact(phi: pd.DataFrame, impact_z: np.ndarray, rcond) -> pd.Series           # Eq. 12
def wrap_up(fit: SparseIPCAResult, shocks: ShockPanel, eval_cfg: EvaluationConfig,
            observables: dict[str, pd.Series] | None = None, phi: pd.DataFrame | None = None) -> WrapUpResult
def retrieval_scores(impact_z: pd.Series, z: pd.DataFrame) -> pd.Series  # I' z(m) per row (day or article)
```

### `oos.py`

```python
def oos_factor(C: np.ndarray, r: np.ndarray, Gamma: np.ndarray, ridge: float = 2.0) -> np.ndarray
def run_oos(panel: IPCAPanel, cfg: PipelineConfig, progress=None) -> OOSResult
    # expanding window per D30-D33; uses tuning.tune when retune_lambda else fixed lam;
    # frozen fit applied to the next refit_every periods; realised OOS MVE Sharpe.
```

### `evaluation.py`

```python
def realized_sharpe(x: pd.Series, annualization: float) -> float
def total_r2(panel, Gamma, F) -> float ; def predictive_r2(panel, Gamma, mu_f) -> float
def price_test_assets(test_assets: pd.DataFrame, factors: pd.DataFrame, t_crit: float, model_name: str) -> PricingTestResult
def grs_test(alphas, residuals, factors) -> tuple[float, float] | tuple[None, None]
def factor_correlations(F: pd.DataFrame, others: pd.DataFrame) -> pd.DataFrame
def placebo_test(shocks, returns, cfg: PipelineConfig, reference_fit: SparseIPCAResult, n: int, seed: int) -> PlaceboResult
def evaluate_run(panel, tuning, fit, oos, wrapup, cfg, test_assets=None, benchmark_factors=None) -> EvaluationReport
```

### `pipeline.py` / `cli.py`

```python
def run_pipeline(attention: AttentionData, returns: ReturnsData, cfg: PipelineConfig,
                 test_assets: pd.DataFrame | None = None, observables: dict[str, pd.Series] | None = None,
                 progress=None) -> PipelineResult
def save_result(result: PipelineResult, out_dir: str | Path) -> dict[str, str]   # file manifest
def load_inputs(attention_path, returns_path, meta_path=None) -> tuple[AttentionData, ReturnsData]  # parquet/csv
# CLI: narrative-ipca run --config cfg.yaml --attention a.parquet --returns r.parquet --out dir
#      narrative-ipca simulate --config sim.yaml --out dir
#      narrative-ipca harness --scenarios baseline,no_factor,topic_null --seeds 3 --out reports/simulation
```

### `simulation.py` / `harness.py`

```python
def simulate(cfg: SimulationConfig) -> SimulatedData
def scenario_config(name: str, base: SimulationConfig | None = None) -> SimulationConfig   # baseline, no_factor, topic_null (alias null), softmax, weak, balanced
def compare_to_truth(result: PipelineResult, truth: SimulationTruth, thresholds: HarnessThresholds,
                     scenario: str) -> HarnessMetrics
def run_harness(cfg: HarnessConfig, pipeline_cfg: PipelineConfig | None = None, progress=None) -> HarnessResult
def write_report(result: HarnessResult, out_dir) -> str   # markdown report path
```

---

## Part D — Simulation data-generating process

Daily calendar: `n_years x 252` business days from 2005-01-03; periods are
calendar months of that calendar. `d` = days in the period.

1. **Factors.** Period-level population moments: `Sigma_ff_period = diag(vol_k^2 / 12)`
   with `vol_k` from `factor_vol_annual`, and `mu_f_period` chosen along the
   direction `Sigma_ff^{1/2} 1` so that `sqrt(12 mu' Sigma^-1 mu) = mve_sharpe_annual`.
   Daily factors: `f_tau ~ N(mu_f_period / d_bar, Sigma_ff_period / d_bar)` with
   `d_bar = n_days / T` the realised mean trading days per period (D48),
   optionally AR(1) with coefficient `factor_ar1` (innovation variance adjusted
   to keep the unconditional variance). `f_period` = sum of daily `f` within
   the period.
2. **States.** `x_tau = f_tau + nu_tau`, `nu_tau ~ N(0, nontradable_share * Sigma_ff_daily)`,
   independent of everything else.
3. **Topics.** `A` (L x K): rows of relevant topics are `signal_strength *`
   standard normal draws (fixed per seed); all other rows zero.
   `z_tau = A x_tau + eta_tau`, `eta` i.i.d. normal (optionally AR(1)) with
   per-topic std `topic_noise_vol * s_l`, `s_l ~ U(0.5, 1.5)`. Placebo topics
   are pure i.i.d. normal with the variance of a randomly chosen relevant
   topic's `z`. Under `signal_strength = 0` ("null") every topic is noise.
4. **Attention levels.** `additive`: `theta_{l,tau} = m_l + s_{l,tau} + z_{l,tau}`,
   `m_l = attention_level_mean * exp(attention_level_dispersion * N(0,1))`
   (normalised so that mean attention sums to one across topics), `s` an AR(1)
   with coefficient `attention_persistence` and innovation std
   `attention_slow_vol * m_l`, with `z` scaled by `m_l` so that shocks are
   proportional to the topic's typical level. `softmax`: the same in log space,
   `theta_tau = softmax(log m + s + z)`, which puts every day on the simplex
   (FASTopic-like) at the cost of a mild non-linearity.
5. **Assets.** Each asset belongs to one class (`AssetClassSpec`), with long-run
   loadings `beta_bar_i ~ N(beta_mean, beta_sd^2)` per factor and idiosyncratic
   annual volatility `idio_vol_annual`. Period loadings follow
   `beta_{i,t} = beta_bar_i + b_{i,t}`, `b` AR(1) with coefficient `beta_ar1` and
   innovation std `beta_innov_sd`. Daily returns
   `r_{i,tau} = beta_{i,t(tau)}' f_tau + eps_{i,tau}`, `eps ~ N(0, vol_i^2 / (12 d_bar))`
   so that the period idiosyncratic variance is exactly `vol_i^2 / 12`
   (Student-t with `fat_tails_df` if set).
6. **Unbalanced panel.** A fraction `unbalanced_fraction` of assets enter or
   exit at a uniformly random date (half enter late, half exit early); a
   fraction `missing_day_fraction` of asset-days are dropped at random.
7. **Truth.** `Gamma_tilde_true = A (A'A)^-1 Sigma_ff_daily^-1`;
   `I_{z->MVE,true} = A (A'A)^-1 Sigma_ff_period^-1 mu_f_period`;
   `sharpe_mve_true = mve_sharpe_annual`; `systematic_r2` = average over assets
   of `beta' Sigma_ff_period beta / (beta' Sigma_ff_period beta + vol_i^2/12)`.

Scenarios (`scenario_config`): `baseline` (defaults); `topic_null` (alias
`null`: `signal_strength = 0`, priced factors kept; report-only, see D47);
`no_factor` (`signal_strength = 0` and every asset class with zero loadings,
`beta_innov_sd = 0`: returns are pure idiosyncratic noise; the chance-level
null); `softmax` (attention on the simplex); `weak` (`signal_strength = 0.35`,
`mve_sharpe_annual = 0.6`); `balanced` (`unbalanced_fraction = 0`,
`missing_day_fraction = 0`). The headline study runs `baseline`, `no_factor`,
`topic_null`, `softmax`, `weak` with three seeds each.

---

## Part E — Harness metrics and expected outcomes

All comparisons are rotation-invariant (D38). `hat` marks estimates.

| Metric | Definition | Expected (baseline) | Threshold |
|---|---|---|---|
| `selection_recall` | selected ∩ relevant / relevant | well below 1: relevant rows of `A` are standard-normal draws, so a share of relevant topics are weak and legitimately not selected | ≥ 0.50 |
| `selection_recall_strong` | recall over the relevant topics whose row norm `||A_l||` is above the median of the relevant rows | near 1 | ≥ 0.80 |
| `selection_precision` | selected ∩ relevant / selected | high under a sparsity-preferring tuning rule; low under the exact in-sample argmax on a flat criterion (D27, Part F.5) | ≥ 0.60 |
| `beta_canonical_corr` | mean over sampled periods of the first canonical correlation between the implied betas `c_{i,t-1} Gamma_hat` and the true `beta_{i,t}` across the assets of the period (rotation-invariant; the object IPCA identifies) | > 0.95 | ≥ 0.90 |
| `placebo_selected` | number of placebo topics selected at tuned lambda (signal scenarios only; under `no_factor` chance-level selection contains placebos at rate `n_placebo / L`, D52) | 0 (App. C.2) under a selective tuning rule; the exact argmax selected 15 of 20 on the flat criterion | ≤ 0 |
| `gamma_subspace_cos` | mean of the informative principal-angle cosines between col(Gamma_tilde_hat) and col(Gamma_tilde_true) on the rows that are relevant *and selected* (needs more than K such rows; with exactly K rows any full-rank block spans R^K) | not identified (D52); observed ≈ 0.6 | reported |
| `factor_canonical_corr` | first canonical correlation between `F_hat` (in-sample) and `f_period_true` | > 0.95 (observed 0.99) | ≥ 0.90 |
| `factor_canonical_corr_mean` | mean over K canonical correlations | reported | — |
| `state_canonical_corr` | first canonical correlation between `x_hat_tau` and `x_true_tau` | depends on `A_hat`, not identified (D52); observed 0.55-0.77 | reported |
| `impact_spearman` | Spearman correlation of `I_{z->MVE}` hat vs true over the relevant topics that were selected (BKS report impact vectors for selected narratives only) | depends on `A_hat`, not identified (D52); observed 0.45-0.65 | reported |
| `mve_sharpe_is` | in-sample MVE Sharpe of the fit | ≈ true (1.0), inflated a little | reported |
| `oos_sharpe` | realised OOS MVE Sharpe | ≈ 0.5–0.9 of true | ≥ 0.5 × true |
| `oos_sharpe_true_mve` | realised OOS Sharpe of the *true* MVE portfolio of true factors | ≈ true | reported (upper bound) |
| `systematic_r2_recovered` | R2 of true systematic return `beta f` on fitted `c Gamma_hat f_hat` (in-sample) | > 0.7 | ≥ 0.50 |
| `total_r2` | model fit | ≈ population systematic R2 | reported |
| `n_selected` | | ≈ n_relevant | reported |
| `instrument_beta_r2_*` | cross-sectional R2 of the instrument `cov[t, :, l]` on the true `beta_t` (mean over sampled periods), split into relevant / noise / placebo topics; `_chance` = K / N | relevant ≫ noise > chance (baseline); noise > chance (topic_null); all ≈ chance (no_factor) | reported |
| `oos_selection_stability` | mean Jaccard similarity of the selected set between consecutive refits | high in baseline *and* topic_null (the spurious instruments persist with the kernel), low only in no_factor; a diagnostic of estimation noise, not of narrative information (D47) | reported |
| `null_selection_lift` (no_factor only) | n_selected / round(0.05 L), the number selected relative to a 5% chance level | ≈ 1, no lift (observed 0.7-0.8 under the argmax and the tolerance rule; LOOCV picks arbitrary points on pure noise) | ≤ 2.0 |
| `null_oos_sharpe_abs` (no_factor only) | \|realised OOS Sharpe\| | within two standard errors of 0 (`2 sqrt(12 / n_oos)` ≈ 0.7 at 90-100 OOS months) | ≤ 0.75 |

Why these expectations: with 20 relevant topics carrying a K=3 structure, 500
assets and 240 months, the KKPS asymptotics (`sqrt(NT)` for `Gamma`) put the
loading error well below the signal; the OOS Sharpe is expected to sit below
the population value because `mu_f` is estimated from at most 20 years of data
(the same mean-reversion caveat BKS's 1.3 carries) and because the 5-day
moving-average shock attenuates the true shock (correlation about 0.91 in the
additive DGP). `no_factor` is the chance-level run: nothing should be
selected beyond chance and the OOS Sharpe should be indistinguishable from
zero. `topic_null` is report-only: by the mechanism of D47 the estimator is
expected to select noise topics above chance, to show
`instrument_beta_r2_noise` well above `K/N`, and to earn a positive but
degraded OOS Sharpe with a stable selected set; the report states these
numbers next to the baseline's. `weak` should degrade gracefully (recall falls, precision
stays, no placebo selected). `softmax` checks that the simplex non-linearity
does not break selection.

Two harness outputs are written per run: `reports/simulation/harness_<date>.md`
(tables, pass/fail, timings) and `reports/simulation/artefacts/` (per-run
metrics JSON, lambda paths, gamma norms).

---

## Part F — Open points and known limitations

1. **Lambda tuning in sample** inherits BKS's optimism (footnote 8); LOOCV is
   implemented but costs `T` fits per grid point. Decision for production:
   `is_sharpe` in the OOS loop, LOOCV as a periodic check (TBC after the pilot).
2. **Sharpe ratios do not certify signal.** Any factor made of risky assets
   inherits a risk premium, and (D47) the kernel-covariance instruments of
   pure-noise topics still span the assets' betas, so Sparse IPCA can build
   priced factor portfolios from topics that carry no information at all,
   and keeps selecting the same ones across refits. The signal-quality
   evidence is therefore relative: the placebo test (real narratives must
   beat variance-matched noise), pricing errors, and the `topic_null`
   comparison in the harness. The harness reports all of them.
3. **Rotation.** Individual factors and states are identified up to an
   orthogonal rotation; only the MVE combination, projections on observables
   and the selected set are interpretable without a convention (D24).
4. **Simplex inputs** make the instruments linearly dependent; harmless for the
   penalised fit, but the unregularised (`lambda = 0`) comparison uses a
   pseudo-inverse and its `Gamma` is not unique. More generally, at the
   dense end of the path (many selected narratives) the Gamma-step Gram is
   near-singular, so individual `Gamma` entries there are identified only
   through fit-level and rotation-invariant quantities (fitted values,
   factors, the MVE); read selected sets and row norms, not entries.
5. **In-sample tuning on a flat criterion.** On the simulated panel the
   in-sample Sharpe barely moves between 10 and 90 selected narratives
   (D27), so the exact argmax is decided by noise and admits many
   uninformative narratives; under pure noise it selects arbitrary points.
   The tolerance rule and LOOCV are the implemented alternatives; which one
   to adopt for production is decided on the pilot data (TBD).
6. **Attention–return simultaneity.** Same-day covariance measures comovement,
   not a one-way response; `attention_lag_days` supports a lagged robustness
   run but structural identification is out of scope.
7. **Scale.** The long-form panel holds `n_obs x (L+1)` floats; for 3,000 assets,
   300 topics and 400 months that is about 3.6 GB in float64. The estimator only
   needs the per-period moments (`PanelMoments`), so a streaming builder is the
   natural next step if that size is reached (TBD).
8. **Not implemented in v0.1:** term-level word clouds (only the vector),
   article-level retrieval beyond `retrieval_scores`, VAR/ICAPM predictive
   regressions (Section 5), Fama-French benchmark download.

---

## Part G — Topic-sensitivity lab and dashboard (code package `exposure_lab`, added 2026-09-29)

The lab is an interactive simulation that shows **how much of an asset's return
variation news topics explain out of sample**. It uses the owner's multi-asset
universe of 55 spread trades with real daily prices for 2015–2025, and either
the 20 manual topics of the AM NLP Sentiment Analytics report (Tables 1–2,
pp. 18–19) or 10–500 generic topics. Topic attention is simulated. The lab
compares two estimators on the same data: the research plan's direct
topic-to-asset sensitivity regression (G.0; the "direct arm" of Comparison
A, research plan v0.3 Section 5) and BKS Sparse IPCA from this package.
G.15 scores them, and the oracle, on the same forecast days with every
method's sensitivities frozen at the training end. BKS enters twice: with the
full covariance history, whose instruments also weigh the days before the
training window, and with the training window only (D88, G.15).

Code: `narrative_ipca/exposure_lab/` (library, no Streamlit imports) and
`dashboard/app.py` (the Streamlit app). Reference data: `data/reference/`.
Market data: `data/market/` (regenerated by `scripts/fetch_market_data.py`).
The BKS core modules are not changed; the lab calls their stages directly.

### G.0 Terms

**Topic sensitivity** (symbol `b_{k,n}`; the matrix `B`, topics x assets):
the expected return response of asset `n` to a one-standard-deviation
attention shock in topic `k`, with the other topics' shocks held fixed. It is
the coefficient in the regression of the asset's return on all topics'
attention shocks at once:

`r_{n,t+l} = a_n + sum_k b_{k,n} s_{k,t} + e_{n,t+l}`

where `r_{n,t+l}` is asset `n`'s return on day `t + l`; `s_{k,t}` is topic
`k`'s attention shock on day `t` (attention minus its mean over the previous
`w` days, divided by its standard deviation); `a_n` is an intercept;
`e_{n,t+l}` is the part not explained by topics; and `l` is the lead (0 =
same day, a co-movement reading; 1 = next day, a predictive reading).

1. **Units**: return per one-standard-deviation shock (% per 1 sd), or
   standardised (the return also divided by its standard deviation), in
   which case `b^2` is about the share of variance the topic explains. The
   lab sets and reports standardised values and offers the % view (G.4,
   G.7.1).
2. **It is not a position size or dollar exposure**: it says how an asset's
   return moves with news attention, not how much of the asset a portfolio
   holds.
3. **In the research plan**, `B = M·L + S` composes it from `M`
   (topic-to-factor sensitivities), `L` (factor loadings) and `S` (the
   sparse asset-specific overlay). This `L` is the plan's factor-loading
   matrix; elsewhere in this document `L` is the number of topics. A
   portfolio's topic sensitivity is the holdings-weighted sum of its assets'
   sensitivities.
4. **In the lab** there are three versions:
   - **set sensitivity** `W`: the values set with the link map and the
     "betas" (G.4);
   - **true sensitivity** `B_true`: the population value the simulation
     produces, including spillovers (G.5.3);
   - **estimated sensitivity** `B_hat`: each method's training-window
     estimate (G.7, G.15).
5. **Symbols in the lab**: the lab's formulas write the observed shock
   `sh_{k,t}` and the intercept `alpha_n` (G.7.1), because `s_{k,t}` there is
   the designed shock of G.5.1 and `a_{k,t}` the attention level of G.5.2.
6. **Naming in code**: the code keeps the older word. In narrative-ipca,
   "exposure" in code identifiers (the package `exposure_lab`,
   `ExposureConfig`, `exposure_heatmap`, `B_hat` and the "exposures" in
   docstrings and variable names) means topic sensitivity.

Renamed from "exposure" on 2026-09-30 (D89) because "exposure" is easily read
as a dollar exposure to the asset.

### G.1 What the lab answers, and what it does not

1. It answers: given a known link structure between topics and assets, how
   much of each asset's return variation do the topics explain in a chosen
   out-of-sample window, how well do the estimators recover the known
   sensitivities, and which topics contributed most to an asset's move.
2. It does not produce evidence about real news. Asset returns are real (or
   artificial), but topic attention is simulated from them. Every number is a
   property of the simulation settings.

### G.2 Universe and data

**Assets.** Two sources:

1. **Listed assets**: the 55 assets of `data/reference/assets.csv`, in the
   order of the source image. Each asset "A v B" is long leg A and short leg
   B (D75); its daily return is `r_A - r_B` (simple daily returns). Outright
   assets (Global Duration, Global Equity, Global Credit, USD) and "XXX v USD"
   pairs have the short leg `cash` with return 0. Legs, their benchmark
   index, the tradeable proxy and the conventions are in
   `data/reference/legs.csv` and `data/market/README.md`.
2. **Generic assets**: `n` synthetic assets (2–500) from the generator of
   G.2.2, labelled `G_ASSET_001, ...`, with a class drawn from
   {Equity, FX, Fixed income}.

**Asset classes** are Equity, FX and Fixed income (rates and credit),
recorded in `assets.csv` with a finer `sub_class`.

#### G.2.1 Market data

Daily history 2015-01-02 to 2025-12-31 on the weekday calendar (Monday to
Friday, 2,869 days). Leg levels are forward-filled for at most five weekdays;
a forward-filled day has return 0 and is flagged stale. Sources: Yahoo
Finance (adjusted close for ETFs, close for indices and FX), FRED H.10 as the
FX fallback. The EM currency leg is an equal-weight, daily-rebalanced basket of
the spot returns of the 12 EM currencies in the list. The per-leg table,
coverage and the QA results are in `data/market/README.md` and
`data/market/manifest.json`.

#### G.2.2 Artificial returns

Used (a) for every asset when the price source is set to "artificial",
(b) for a listed asset whose leg failed to download entirely (the asset is
flagged `source = artificial` in the asset table), and (c) for generic assets.
Model, with `t` the day and `n` the asset:

`r_{n,t} = lambda_n' F_t + e_{n,t}`

where `F_t` are three latent daily market factors (risk-on, US dollar, rates),
each a GARCH(1,1) process (`omega` set so that the annualised volatility is
10%, `alpha = 0.08`, `beta = 0.90`); `lambda_n` are class-specific loadings
drawn once per asset (Equity `(0.6, -0.1, -0.1)`, FX `(0.3, -0.6, 0.0)`,
Fixed income `(-0.2, 0.1, 0.5)`, each plus `N(0, 0.2^2)`); and `e_{n,t}` is
Student-t (5 degrees of freedom) idiosyncratic noise scaled so that the
asset's total annualised volatility equals the class target (Equity 10%,
FX 9%, Fixed income 5%, with a per-asset multiplier `U(0.6, 1.5)`). When the
factor part alone exceeds 90% of the target variance (typical for Fixed
income), the asset's loadings are shrunk until it equals 90%, so the total
volatility still hits the target. Seeded per asset id, so an asset's series
does not depend on which other assets are in the universe.

### G.3 Topics

1. **Manual topics** (`data/reference/topics.csv`): the Sector Ontology
   `S1-S11` (group Sector) and the Global Multi-Asset Hierarchy `A1-A6`
   (group Macro) and `B1-B3` (group Micro), with names and scope as printed
   in the report. The topic set is one of: none, sector (11), economic (9),
   both (20).
2. **Generic topics**: `n_generic` topics (0–500) labelled `G001, ...`,
   group Generic. A share `generic_signal_share` of them carry links (G.4);
   the rest are pure noise (placebos).
3. The total number of topics `L` is the manual count plus `n_generic`; it
   must be between 1 and 520. The dashboard offers 10–500 generic topics when
   no manual set is chosen.

### G.4 Link map and set sensitivities ("betas")

**Link map.** A table of `(topic, asset, tier, sign, mechanism)` rows. `tier`
is strong, moderate or weak; `sign` is +1 or -1; `mechanism` is one sentence
stating why the pair is linked. Pairs not in the table are not linked.

1. **Default map for manual topics x listed assets**:
   `data/reference/link_map.csv`, authored for this lab (about 100 rows). It is
   illustrative, not a research claim: it encodes a plausible economic story
   so that the simulation has recognisable structure. The sign convention is
   the direction the asset moves when attention to the topic rises, under a
   stated reading of each topic (for example, attention to Financial
   Conditions & Market Stress rises in stress episodes). The dashboard shows
   the table and allows edits for the session.
2. **Random map** for every other combination (generic topics, generic
   assets, or manual topics with generic assets): each linked topic gets 1
   strong, 2 moderate and 3 weak links to assets drawn without replacement
   (capped at the number of assets), signs +1 or -1 with equal probability,
   seeded by the link seed (`ExposureConfig.seed`; the topic noise has its own
   seed, D71).

**Set sensitivities.** The user sets `n_betas` (the number of set
sensitivities, or betas) in {1, 2, 3} and up to three values:

| `n_betas` | strong | moderate | weak |
|---|---|---|---|
| 1 | `beta_1` | `beta_1` | `beta_1` |
| 2 | `beta_1` | `beta_2` | `beta_2` |
| 3 | `beta_1` | `beta_2` | `beta_3` |

The values are in **standardised units**: a topic linked to a single asset
with value `beta` has attention shocks whose correlation with that asset's
return is `beta` (before the attenuation of G.5.3), so the share of the
asset's variance that topic explains is about `beta^2`. Defaults
`beta_1 = 0.35`, `beta_2 = 0.15`, `beta_3 = 0.05` reuse the plan's
strong/moderate/weak magnitude buckets (background doc, elicitation section)
as correlation units. The `%` view converts a standardised sensitivity to a
return response in percent per one-standard-deviation shock by multiplying
with the asset's daily volatility (in percent).

The **set-sensitivity matrix** `W` (`L x N`, built by `links.design_matrix`)
has `W_{k,n} = sign_{k,n} * beta(tier_{k,n})` for linked pairs and 0
elsewhere.

### G.5 Data-generating process: topics built from prices

Owner decision 2026-09-29: asset returns stay as they are (real or
artificial) and each topic's attention is built from the returns of its
linked assets plus noise. Symbols:

- `n = 1..N` assets, `k = 1..L` topics, `t` trading days of the weekday
  calendar.
- `r_{n,t}`: the asset's daily return; `mu_n`, `sigma_n`: its full-sample mean
  and standard deviation.
- `rt_{n,t} = (r_{n,t} - mu_n) / sigma_n`: the standardised return, clipped at
  plus or minus 8 in the construction only (so a single extreme day such as
  2015-01-15 in CHF cannot dominate a topic), missing values set to 0.
- `R`: the `N x N` full-sample correlation matrix of the standardised
  returns (pairwise complete).
- `l` (lead) in {0, 1}: topic attention on day `t` relates to returns on day
  `t + l`. `l = 0` is the same-day (risk) reading; `l = 1` the predictive
  (signal-grade) reading of the plan.

#### G.5.1 Designed shocks

`s_{k,t} = W_k rt_{t+l} + sigma_{u,k} u_{k,t}`

In words: topic `k`'s designed attention shock on day `t` is the sum of its
linked assets' standardised returns on day `t + l`, weighted by the set
sensitivities `W_k`, plus independent news noise. `W_k` is row `k` of `W`, `rt_{t+l}` the
vector of standardised returns, `u_{k,t}` i.i.d. Student-t noise with
`noise_df` degrees of freedom scaled to unit variance (seeded by the noise
seed `ExposureConfig.noise_seed`, which also draws `m_k` and `g_{k,t}` of
G.5.2), and
`sigma_{u,k}^2 = 1 - W_k R W_k'` so that `Var(s_k) = 1` in population. For
days where `t + l` is beyond the sample, `rt` is 0 (noise only).

**Feasibility.** If `W_k R W_k' > 0.95`, row `W_k` is multiplied by
`sqrt(0.95 / W_k R W_k')` and the scaling is recorded and shown. This happens
only when a topic is linked with large values to many positively correlated
assets.

#### G.5.2 Attention levels

`a_{k,t} = m_k + g_{k,t} + kappa * s_{k,t}`, clipped below at `1e-6`

In words: the attention level is a topic-specific base level `m_k ~ U(0.15, 0.35)`,
plus a slow persistent component `g_{k,t}` (AR(1) with coefficient `0.995`
and innovation standard deviation `0.1 * kappa`, started from its stationary
distribution), plus the scaled designed shock with `kappa = 0.02`. The scale
mimics the report's relative-attention series (levels 0.15–0.35, daily
deviations of a few hundredths). The share of clipped cells is recorded.

#### G.5.3 What the estimators observe, and the population truth

The estimators see only the attention levels. The observed shock is the
package's D9 shock with window `w`:

`z_{k,t} = a_{k,t} - (1/w) sum_{j=1..w} a_{k,t-j}`

standardised per topic by its standard deviation on the training window,
`sh_{k,t} = z_{k,t} / sd_train(z_k)` (the plan's shock definition, with the
trailing standard deviation replaced by the training-window one so that the
standardisation is out-of-sample safe).

Because the trailing mean contains earlier designed shocks, `z` is a noisy
reading of `s`. With `v_k = Var(z_k) = kappa^2 (1 + 1/w) + Var(g_t - (1/w) sum_j g_{t-j})`,
the **attenuation** of topic `k` is `a_k = kappa / sqrt(v_k)`, about
`1/sqrt(1 + 1/w)` (0.91 for `w = 5`, 0.71 for `w = 1`).

The **population truth** on the observed standardised shocks. Write
`p_t = W rt_c_{t+l}` for the noise-free part of the designed shock and
`q_t = p_t - (1/w) sum_{j=1..w} p_{t-j}` for the same trailing-mean filter
applied to it. The observed shock is `z_t = kappa q_t + kappa e_t + D_t`, with
`e_t` the filtered news noise (variance `(1 + 1/w) sigma_u^2`, independent of
everything else) and `D_t` the filtered slow component. Then:

- `Var(z) = kappa^2 Cov(q) + kappa^2 (1 + 1/w) diag(sigma_u^2) + Var(D) I`,
  and `S_z` is `Var(z)` scaled to a unit diagonal: the correlation matrix of
  the observed shocks.
- `C = kappa Cov(q_t, rt_{t+l}) / sd(z)` (`L x N`): the covariance of the
  observed standardised shocks with the standardised returns.
- `B_true = S_z^{-1} C` (`L x N`): the **true sensitivity** of asset `n` to
  topic `k`, in standardised units: the multivariate regression coefficient
  of the standardised return on the observed standardised shocks.
- `R2_true_n = C_n' S_z^{-1} C_n`: the population share of asset `n`'s
  variance the topics explain (`C_n` the `n`-th column of `C`).

The moments are the full-sample moments of the filtered signal, so the truth
keeps the serial correlation of daily returns. This matters on the real data:
spreads whose legs close at different times have lag-1 autocorrelations down
to -0.5 (Quality v World EQ), and the first closed form of this section, which
assumed serially uncorrelated returns (`C = diag(a) W R`,
`S_z = diag(a)(1 + 1/w)(W R W' + diag(sigma_u^2)) diag(a)`), missed the truth
by more than 0.03 on such assets in the Monte Carlo test. With serially
uncorrelated returns the two coincide. `R` in G.5.1 is implemented as `V`, the
covariance of the clipped, zero-filled standardised returns, which equals `R`
when no day is clipped or missing.

**Missing returns (D78).** For an asset observed on only some days, `C` and
the `Cov(q)` inside its `S_z` are taken over its observed days, while the
shocks keep their all-day standardisation. `B_true` is then the regression
coefficient of the asset's standardised return on the standardised shocks
over its observed days, which is what the direct estimator targets. Assets
with the same observation mask share one solve; with complete data (the
committed store has no missing cell) there is one group and the formulas
above apply as written. With Global Equity's first 1,200 days removed, a Monte
Carlo check gives an R2 of 0.46; the truth was 0.58 before this rule and is
0.45 with it.

**A key property: the truth has spillovers.** A topic linked only to the
Energy spread also co-moves with every asset correlated with the Energy
spread, so `B_true` is not sparse even though `W` is. The dashboard shows the
set sensitivities (`W`), the true sensitivities (`B_true`) and the estimates
side by side.

### G.6 Time windows

- **Training window** `[train_start, train_end]` (library defaults 2015-01-02
  and 2022-12-30), at least 21 weekdays, about one month (D81). Below about a
  year the dashboard shows the cost in plain numbers: a sensitivity's standard
  error is about `1/sqrt(n)` in standardised units (`n` the training
  weekdays; 0.21 at one month against about 0.32 for a strong link), and the
  default elastic-net penalty `sqrt(2 ln L / n)` sets most sensitivities to zero.
  BKS needs at least 24 training weeks on its own (`bks.MIN_TRAIN_PERIODS`,
  D84) and refuses shorter windows.
- **Near the data start** (D87): the 21-day floor counts weekdays with a
  topic shock, and the first `w` days of the data have none (the shock needs
  `w` earlier days), so the earliest cut-off is 2015-02-06 at `w = 5`
  (2015-02-02 at `w = 1`, 2015-02-27 at `w = 20`); the cut-off control
  starts there. BKS with the full history also skips its first 52 weeks of
  data (burn-in, D17): with inverse-volatility weighting its first usable
  week ends on 2016-04-08, and `bks.training_weeks` counts only the weeks
  after it. The training-window variant (D88) has no burn-in; it loses the
  first one or two weeks of the window instead (24 of the 26 week ends of the
  dashboard default).
- **Forecast window**: starts at `forecast_start` (library default
  2023-01-02, must be after `train_end`; a gap is allowed) and lasts
  `forecast_weeks` weeks (1 to 12), i.e. the weekdays in
  `[forecast_start, forecast_start + 7 * weeks)`.
- A return day `t + l` is in a window when its date is; the matching shock
  day is `t`.
- Everything fitted uses training data only: `sd_train(z)`, return
  standardisation for the estimators, estimated sensitivities, BKS `Gamma`. The forecast
  window uses realised shocks and returns with frozen training estimates.

**The dashboard's Time windows panel** (owner request 2026-09-29, D84). The
training window is set by its end and its length, not by two dates:

1. **Training end (cut-off)**: the last day of the training window.
2. **Training length**: 1, 2, 3, 4, 6, 9, 12 or 18 months, or 2 to 10 whole
   years, counted back from the cut-off in calendar months.
3. The start is derived: `train_start = (train_end + 1 day) - length`. A
   month-end cut-off therefore gives whole calendar months: 2025-06-30 with
   6 months gives 2025-01-01 to 2025-06-30 (129 weekdays).
4. Two corrections, each stated in the caption under the controls:
   - when the window has fewer than 21 weekdays (a February month has 20),
     the start moves back to the 21st weekday before the cut-off (2025-02-28
     with 1 month gives 2025-01-31 to 2025-02-28);
   - when the start falls before the first day of the data, it is clipped to
     2015-01-02 (2020-06-30 with 10 years gives 2015-01-02 to 2020-06-30,
     1,433 weekdays).
5. **Forecast start** and **forecast length** (1 to 12 weeks) are set as
   before.

Dashboard defaults (owner, 2026-09-29): cut-off 2025-06-30, 6 months;
forecast start 2025-07-01, 4 weeks (20 return days, 2025-07-01 to
2025-07-28). The library defaults of `WindowConfig` are unchanged, so scripts
and tests keep the eight-year training window. The default cut-off is a
Monday: BKS's weekly periods put 2025-06-30 in the first forecast week and in
no training week, and the BKS tab notes it (G.7.2).

### G.7 Estimators

#### G.7.1 Direct sensitivity regression (the plan's direct arm)

Per asset `n`, on the training days:

`rh_{n,t+l} = alpha_n + sum_k b_{k,n} sh_{k,t} + e_{n,t+l}`

where `rh` is the return standardised by the training mean and standard
deviation, `alpha_n` an intercept and `b_{k,n}` the topic sensitivity (G.0).
Methods:

1. `elastic_net` (default; the plan's baseline estimator): scikit-learn
   `ElasticNet` with `l1_ratio` (default 0.9) and penalty `alpha` chosen by
   `penalty`: `universal` (default) `alpha = sqrt(2 ln L / n_train)`, the
   universal threshold of lasso theory, which keeps the expected number of
   noise topics near zero and scales with the topic count; `fixed` (a
   slider); `cv` (time-series cross-validation, 5 blocked folds, slower).
2. `ridge`: closed form; `lambda` by generalised cross-validation over a
   30-point grid when not fixed, with the intercept counted in the degrees
   of freedom (D86).
3. `ols`: least squares (refused when `L >= n_train / 2`).
4. `oracle`: `b = B_true` (G.5.3), with the same training scales as the
   estimators. What an estimator reaches when it recovers the true
   sensitivities exactly; a reference, not an estimator. It reproduces the
   oracle line of G.8 exactly (D74).

Sensitivities are reported in standardised units and in percent per
one-standard-deviation shock (`b * sd_train(r_n) * 100`, with the asset's
training volatility for the estimated, the true and the set sensitivities
alike, D74). A pair is
**selected** when `b_{k,n} != 0` (sparse methods) or `|b_{k,n}| >= tau`
(dense methods; `tau = 0.05`).

#### G.7.2 BKS Sparse IPCA

The lab calls the package stages with weekly periods (`period = "W"`):
`align_inputs -> attention_shocks (window w) -> build_covariance_panel -> build_panel`,
then fits on the periods whose last trading day is on or before `train_end`,
and evaluates the weekly periods whose last trading day is in the forecast
window with `oos.oos_factor` on each period's cross-section. Lab defaults
differ from `configs/default.yaml` where weekly periods require it: kernel
half-life in months converted to a weekly `xi = 0.5^(1 / (half_life * 52/12))`
(69 months gives 0.9977), `burn_in_periods = 52`, `asset_weighting = "inverse_vol"`
(D7; the universe mixes 3% and 20% volatility assets), `attention_lag_days = l`.
User-settable: `K` (1–6, default 3, and below the number of assets), half-life,
the lambda rule (`tolerance`, default 2% as in D51; `argmax`, BKS exact;
`fixed`, where 0 fits plain IPCA), the grid (`n_lambdas` default 12,
`lambda_ratio` default 1e-2), and the intercept penalty. Runs on request and
is cached; above 100 topics the dashboard keeps the coarse grid and warns
about runtime (Section G.10). It needs at least 24 training weeks (D84). The
method comparison turns the training fit into topic sensitivities (G.15).

**Covariance history (D88).** `BKSLabConfig.history` chooses the data the
panel is built from:

1. `"full"` (default, as in BKS): every day from the start of the data. A
   training week's instruments weigh the whole history before it; the first
   52 weekly instrument periods are a burn-in (D17), an instrument needs 60
   observed days, and `inverse_vol` divides each daily return by its trailing
   252-day volatility.
2. `"training"`: the training window only. Returns from `train_start` on and
   attention from `w` weekdays before `train_start` on, so the first shock
   falls on `train_start` and the shocks are the direct estimator's `z`.
   Nothing before `train_start - w` weekdays enters any BKS quantity. Three
   settings differ:
   - `inverse_vol` divides by the asset's training standard deviation
     (population, over the return days up to `train_end` that pair with a
     shock: the direct estimator's `sd_train(r_n)`), the same divisor on
     every day and frozen for the forecast weeks. A trailing volatility
     would need 63 days of warm-up inside the window (a quarter of 252, the
     package rule), which leaves about 12 of the 26 weeks of a six-month
     window. The divisor uses training days after the week it scales; that
     is in-sample for the training fit and uses nothing after the cut-off.
     It also makes the unit conversion of G.15 exact.
   - The burn-in is `burn_in_weeks_training = 0` weeks.
   - An instrument needs `min_days_training = 3` observed days.
3. **The trade-off:** the training variant's first instruments are
   covariances over a few days to a few weeks of data, far noisier than the
   full history's (at least 52 weeks). With these settings the dashboard
   default window (2025-01-01 to 2025-06-30) keeps 24 of its 26 week ends,
   because the first week's instrument has only 2 days. 3 days is the largest
   minimum that keeps every six-month window with a month-end cut-off from
   2016 to 2025 at 24 weeks or more for both leads; 5 days (one trading week)
   would refuse 3 same-day and 12 next-day windows of the 119.
4. Forecast weeks (the BKS tab's own OOS R2): the returns after `train_end`
   stay in the panel, and their instruments accumulate from `train_start` up
   to the week before (no look-ahead).
5. The training-history panel holds one training window's data and scales:
   its cache key holds the window (G.10), and `fit_bks` refuses another
   window on it.

`bks.training_weeks` and `bks.kernel_history_share` handle both histories
without building a panel; under `"training"` the share is 0.

**What the BKS OOS R2 measures (D79).** Each forecast week's `K` factors are
estimated from that week's own returns (`oos_factor`), so the BKS OOS R2 is a
contemporaneous factor fit. It is not comparable to the direct estimator's
R2, which fits nothing in the window: with all set sensitivities (betas) at 0
the direct R2 is 0
while BKS still shows about 24–28%, because noise topics' instruments inherit
the assets' betas (D47). The lab therefore reports, next to the pooled OOS
R2, the same R2 with the topic instrument columns shuffled across the week's
assets (20 seeded shuffles per week): what `K` weekly factors reach with
instruments unrelated to the assets (0.11 against 0.33 at the defaults).
Three guards: `K` at or above the number of assets is refused (each week's
factors would fit its returns exactly); at `lambda = 0` the OOS factor uses
ridge 0, matching the unpenalised factor step of plain IPCA (D18), and every
other `lambda` ridge 2; a forecast window that starts mid-week records both
the days before `forecast_start` in the first week and the window days in a
week that ends after the window, and the BKS tab shows the span it scores.

Per-asset fitted return in forecast week `t`:
`rf_{n,t} = c_{n,t-1} Gamma f_t = Gamma_0 f_t + sum_l cov_{n,t-1,l} Gamma_l f_t`.
The per-topic terms sum exactly to the fitted value, but by D52 only
`c Gamma` is identified, so the split across topics belongs to the sparse
representative the lasso chose. The dashboard labels it accordingly.

### G.8 Evaluation in the forecast window

Symbols: `H` the set of return days in the window, `rhat_{n,t+l} = sd_train(r_n) * sum_k b_{k,n} sh_{k,t}`
the topic-explained return (no intercept). The **oracle** is the same formula
with `b = B_true` and the same training scales `sd_train(r_n)` and
`sd_train(z_k)`, so no forecast-window data enters it (D74). The evaluation
refuses a forecast window that starts on or before the training end (D65).

1. **OOS R2 (explained variation)**: `R2_n = 1 - sum_H (r - rhat)^2 / sum_H r^2`,
   uncentered (zero-mean benchmark, the IPCA total-R2 convention). Reported for
   the estimator and the oracle. Can be negative.
2. **OOS correlation** of asset `n` with topic `k`: the Pearson correlation
   over `H` of `r_{n,t+l}` and `sh_{k,t}` (needs at least 3 days).
3. **Contribution (return attribution)**: `c_{k,n} = sum_H sd_train(r_n) * b_{k,n} * sh_{k,t}`,
   in return points over the window; the realised move is `sum_H r_{n,t+l}`
   and the residual "not explained by topics" is the difference. The
   contributions and the residual sum exactly to the realised move. The same
   with `B_true` and the same `sd_train(r_n)` gives the true contribution.
   **A key limitation (D76):** each shock is attention minus its trailing
   mean, so `sum_H sh_{k,t}` telescopes to the attention level at the
   window's edges and largely cancels. Over 39 four-week windows at the
   defaults, the correlation of the window sums of realised and oracle
   explained returns is 0.18 against 0.47 for daily returns, and the
   Energy spread's top contributor is S2 Materials, not S1 Energy. The
   return attribution therefore understates the topics' role.
4. **Variance share** (the dashboard's default view, D76):
   `v_{k,n} = sum_H (sd_train(r_n) b_{k,n} sh_{k,t}) r_{n,t+l} / sum_H r^2`:
   the share of the window's (uncentered) return variation that co-moves with
   topic `k`'s explained part; the shares sum to `sum_H rhat r / sum_H r^2`.
   It uses every day of the window (for the Energy spread at the defaults,
   S1 Energy on top at 19.7%, true 19.9%).
5. **Recovery** (training fit against truth, over all topic-asset pairs):
   sign agreement on the design-linked pairs that the estimator selected;
   coverage (share of design-linked pairs selected); Matthews correlation of
   "selected" against "truly sensitive" (`|B_true| >= tau`); Spearman rank
   correlation of `b` with `B_true`; RMSE. These mirror the plan's golden-set
   scoring (sign agreement, rank correlation, MCC).
6. **Window sweep**: the median OOS R2 over consecutive non-overlapping
   windows of the chosen length from `forecast_start` to the end of the data
   (at most 150 windows; an incomplete last window is dropped), with the
   training fit frozen. The dashboard caption states the first and last day
   covered, and says so when no complete window fits.

**Why not a regression inside the forecast window** (owner question,
2026-09-29). A regression of the window's returns on the topics is in-sample
for that window and has 5 to 60 daily observations against 9 to 520 topics:
it is either not identified or fits noise. The contribution in point 3 uses
sensitivities estimated on the training window and frozen, which is the plan's
attribution chain (research plan v0.3 Section 6). One-topic-at-a-time
regressions double count correlated topics and do not add up; Shapley or LMG
decompositions of R2 are order-free but combinatorial (sampled at 500 topics)
and answer a variance question rather than "what moved the asset"; the BKS
per-topic split is not identified (D52). Point 4 is the default view and
point 3 the alternative (D76); the dashboard's "Why this method" note gives
this reasoning in the app.

### G.9 Dashboard (`dashboard/app.py`)

Three pages in the top navigation (D82, D85, D90): **Simulation lab**
(everything below), **BKS trace** (the BKS run of the current settings traced
step by step, G.16) and **Real data** (the placeholder of G.14). The sidebar
is shared by all pages: it is drawn and validated before the chosen page
runs, and the Real data page lists the settings that will apply to real data
(D85). Run BKS works on the Simulation lab and BKS trace pages, which handle
its requests the same way and keep the same result; on the Real data page the
sidebar's Run BKS button is disabled.

Sidebar controls, grouped:

1. **Universe**: listed or generic assets; subset of listed assets; price
   source (real or artificial); number of generic assets.
2. **Topics**: manual set (none, sector, economic, both); number of generic
   topics; share of generic topics with links.
3. **Sensitivities** (named "Exposures" until 2026-09-30, D89): the
   number of set sensitivities (betas, 1–3); their values, one per link
   tier (with a note of any feasibility scaling in the run and the largest
   set sensitivity 1 that needs none); lead (same day or next day); topic
   noise tails; link seed and noise seed.
4. **Time windows** (named "Windows" until 2026-09-29, D84): training end
   (cut-off) and training length (1 month to 10 years), with the resulting
   window and any correction in a caption (G.6); forecast start; forecast
   length in weeks (1–12); shock window `w`.
5. **Direct estimator**: method, penalty rule and values; a run-time warning
   when cross-validation would take more than about 5 seconds (about 25 s at
   520 topics x 55 assets).
6. **BKS model**: covariance history (a radio, "Full history before the
   cut-off" or "Training window only", D88; it drives the BKS tab and this
   Run BKS button), `K`, kernel half-life, lambda rule, grid, intercept
   penalty; a "Run BKS" button, disabled when BKS cannot use the training
   window (the BKS tab's check, `_ui.bks_training_check`, D87).

Main tabs:

1. **Overview**: headline figures (median OOS R2 for the estimator and the
   oracle, population R2, recovery metrics), OOS R2 per asset, the window
   sweep. Notes: feasibility scaling per topic in plain numbers (factor,
   largest link before and after); the medians over the linked assets when
   some assets have no link.
2. **Correlation table** (named "Exposure table" until 2026-09-29): the
   asset x topic heatmap in the layout of the
   owner's example (assets as rows, topics as columns, blank cells, an
   AVERAGE row). Cell metric: OOS correlation (default), estimated sensitivity,
   true sensitivity, set sensitivity (`W`), OOS contribution. Blank rule, which
   follows the view (D69, amended 2026-09-30): the estimated sensitivity, the
   OOS correlation and the OOS contribution blank the pairs the estimator did
   not select; the true sensitivity blanks the pairs with `|B_true| < tau`
   (standardised units, whatever the unit shown); the set sensitivity blanks
   the pairs with no link. The checkbox label and the chart subtitle name the
   rule in force, and a `|value|` threshold in the units shown applies on
   top. The AVERAGE
   row averages over the displayed rows with blank cells counted as zero (the
   convention the example follows). Optional long/short view per asset flips
   the row sign and prefixes "L" or "S", as in the example's key-view
   column. Columns: manual topics in ontology order, then generic topics by
   average absolute value; at most a user-chosen number of columns. Colours:
   red and blue (default) or the example's red and black; axis titles
   "Asset (key view)" and "Topics".
3. **Topic contributions**: pick an asset; variance share (default) or
   return attribution; horizontal bars of the topics' values, largest
   absolute value on top, with the true value overlaid and an "other topics"
   bar; the residual and the realised move (or total variation) in a second
   panel with its own scale, so the topic bars stay readable; units pp for
   returns and % for shares; optional roll-up by topic group; the cumulative
   realised versus explained return through the window; a "Why this method"
   note (G.8); the simulated attention of up to three topics with the largest
   non-zero return-attribution contributions. When no topic contributes
   (every estimated sensitivity of the asset is zero, as when the elastic net
   selects no topic for it), a note replaces the attention chart and the
   "Largest topic" tile reads "none" (2026-09-30).
4. **Compare methods** (G.15, D83, D88): the methods to compare (default
   elastic net, ridge with GCV, BKS-implied (full history), BKS-implied
   (training window) and the oracle; OLS on request) and a
   summary table with one row per method, the oracle last as the reference;
   an unavailable method stays in the table with its reason. The table puts
   the three out-of-sample columns next to the method name and the recovery
   columns after them, and shows a dash where a figure does not apply. The
   lead caption says that BKS-implied (full history) sees more history than
   the direct methods (with the kernel-weight share before the training
   start), that BKS-implied (training window) is the like-for-like variant,
   and that the BKS tab's own OOS R2, named with its value after a run, is
   not the comparable figure. Two charts: the OOS R2 per asset in this window for
   every method (a dot plot, assets sorted by the oracle's R2, one colour and
   marker symbol per method, the oracle as an ink tick, values below -50%
   drawn at -50%), and the median OOS R2 across consecutive forecast windows
   (one line per method, same colours, values below -100% drawn at -100%).
   Each method keeps its colour whatever the selection (slots in
   `compare.METHODS` order). An inspect section for one method (default the
   sidebar's direct method, followed until the user picks another; the
   oracle is the reference and not an option): recovery tiles, estimated
   against true sensitivities, and OOS R2 per asset against the oracle, with the
   D52 caveat for BKS-implied, its covariance history and the number of
   factor directions used when `Gamma_tilde` has rank below `K`, and a link
   to the BKS trace (G.16; for the variant whose history the sidebar does not
   show, a button that sets the sidebar's covariance history and opens the
   trace).
   Each BKS-implied variant needs a BKS run of this browser session on the
   current settings and its history (D80); without one the tab says why, per
   variant, and offers one Run BKS button that fits every selected variant
   not fitted yet. It never starts a BKS fit on its own. When a BKS-implied
   method is selected, a caption under the table points to the reasons it
   scores lower (Data and method tab, G.15).
5. **BKS** (the sidebar's covariance history, named in the caption):
   selected topics and their `Gamma` row norms, the lambda path, the
   pooled OOS R2 next to the shuffled-instrument reference (D79), OOS R2 per
   asset against the current direct fit (labelled as not the same measure,
   with the days each one scores), and the per-topic split for the chosen
   asset with the D52 caveat. A link under the Run BKS button opens the BKS
   trace of the same run (G.16, D90).
6. **Lists**: the 55 assets (with legs, index, proxy, data source), the 20
   manual topics (ID, group, name, scope), and the link map (editable for the
   session, with a reset).
7. **Data and method**: sources, assumptions (TBC items), why BKS-implied
   scores lower than the direct methods (a short version of G.15.1; the
   Compare methods tab points to it), limitations (Section G.12).

**"How to read" captions** (owner request 2026-09-30). Every chart, results
or reference table and row of tiles, on every page, has a caption right under
it (the link map's sits directly above its editor, which the edit buttons
follow): a lead line ("How to read the tiles:", "How to read this chart:",
...), then one bullet per item, and each bullet ends with one sentence
"Example: ...". Diagnostics (the stage timings) and input editors (the
long/short view, the session's link edits) have none. The Topic
contributions tab's "How to read the two views" block is the model: its
bullets keep the owner's wording and only gain the example. Rules:

1. The examples use static, illustrative numbers, never figures computed from
   the current run. They form one consistent set for the whole dashboard:
   Energy Global v World EQ with a training volatility of about 1.1% a day;
   its strong Energy link set at 0.35, with a true sensitivity of 0.35 and an
   estimate of 0.32 standardised; shocks summing to +2.0 over the window
   (1.1% x 0.32 x 2.0 = +0.70 pp; +0.77 pp with the true sensitivity); a
   realised move of +1.20 pp; 20 return days in the four-week default window;
   and 55 assets x 20 topics = 1,100 pairs. Figures quoted from the default
   run (for example medians of 17% and 23%) were checked on the dashboard
   defaults. Examples that count days assume the same-day lead.
2. Every example follows the code. The owner's sample wording ("a variance
   share of 10% for a topic means it explains a 3% change") does not: the
   variance share is a share of the window's uncentered squared-return
   variation (G.8 point 4) and has no percentage-point value. The example
   says instead that a topic has a 10% share when its explained return is one
   tenth of the asset's return on every day of the window.
3. The texts live in `dashboard/_ui.py` (`how_to_read` and the `HOW_*`
   constants and `how_*` builders); a text part that depends on the settings
   (the kernel half-life, the number of windows) is filled in, the example is
   not. The Correlation table's metric caption shows the chosen metric's
   bullet (and the unit bullet for the three sensitivity metrics), so it
   follows the view as the blank rule does; the contributions tiles' caption
   follows the chosen view, and the roll-up caption shows when the roll-up is
   on.
4. Plain words only: no decision numbers or other design-register codes in
   these captions.

### G.10 Performance and caching

Stages are pure and keyed by the hash of the sub-configuration they depend on
(`LabConfig.key(stage)`): market data (universe), simulation (universe,
topics, `ExposureConfig` with the set sensitivities, links, lead and seeds,
attention), observed shocks (plus `w` and the training window), direct fit
(plus the estimator), window evaluation (plus the forecast window), BKS panel and BKS fit. Two stages serve the method comparison
(G.15): `bks_implied` (keyed by the BKS fit, the observed shocks and `tau`)
and `comparison` (keyed by the evaluation, the methods and one BKS token per
BKS-implied method that records whether and which fit was scored). The BKS
panel key holds the covariance history and its settings, and under the
training history also the training window (D88). Changing only the
forecast window re-runs only the evaluation, and the comparison re-scores
the cached fits without refitting: about 0.05 s at 20 topics and 0.5 s at
520 topics including the ridge and OLS fits. Measured on 2026-09-29 (55
listed assets, 11 years): the
simulation and truth under a second at 20 and 500 topics; the direct elastic
net with the universal penalty, evaluation and sweep under a second at 500
topics (cross-validation about 25 s at 520 topics); BKS weekly under a second
at 20 topics (panel 0.3 s, fit 0.4 s) and about 36 s at 500 topics on the
coarse grid (the default grid took 9 minutes monthly).

The dashboard's session is shared by every browser session of the server
(D80): each stage key is computed once (a second caller waits on a per-key
lock), stage timings are kept per thread, the BKS stages keep at most two
results each, and the cached BKS panel drops the daily shocks and the 3-D
covariance array (about 1.1 GB at 500 assets x 500 topics).

The BKS trace (G.16) is the stage `bks_trace`, keyed by the `bks` key and
cached like the BKS stages (two results). It reads the cached panel and fit
and never starts a fit. It recomputes the daily shocks, and the covariance
instruments only for the selected asset (one asset's column of the
recursion is exact on its own), never the whole 3-D array. Measured
2026-09-30: 0.3–0.5 s on the dashboard defaults including a refit of the
twelve-point lambda path (skipped above 60 topics; the `lambda_max` recompute
above 100 topics), 1.4 s at 200 assets x 120 topics, 2–10 s at 500 x 500
(dominated by dense 500 x 500 decompositions); the per-selection detail of a
step (asset, topic, week) takes 0.03–0.25 s and is computed only for the step
on show.

### G.11 Decisions (D53–D72 and D74–D90)

D73 is the Part B decision on `.npz` timestamps of the same day; the lab's
decisions continue at D74.

- **D53 Location and scope.** Owner decision 2026-09-29: the lab lives in this
  repository as `narrative_ipca.exposure_lab` plus `dashboard/`, with
  reference data in `data/reference/` and market data in `data/market/`.
  Core estimator modules are unchanged; the lab composes their stages.
- **D54 Asset return definition.** Spread return = long-leg minus short-leg
  simple daily return; outrights against cash. Equity legs are USD-listed
  total-return ETFs; rates legs are local-currency returns against the
  EUR-hedged global government index; FX legs are spot (no carry). All
  assumptions are listed with (TBC) in `data/market/README.md`.
- **D55 Market data store.** `data/market/` is the single store (committed;
  small), regenerated by `scripts/fetch_market_data.py`. A listed asset
  whose leg failed is filled with artificial returns (G.2.2) and flagged.
- **D56 Artificial returns** follow G.2.2: a three-factor GARCH model with
  class-specific loadings and volatility targets; used for generic assets,
  the artificial price mode and failed legs.
- **D57 Topic sets.** The 20 manual topics are transcribed from the report
  (Tables 1–2); generic topics are unnamed and a configurable share carries
  links.
- **D58 Link map.** Three tiers with sign and mechanism; the default map for
  manual topics x listed assets is authored and illustrative; everything else
  uses the seeded random map of G.4.
- **D59 Sensitivity units.** Values are set in standardised (correlation)
  units because the universe mixes volatilities from 3% to 20%; one raw
  percentage value would explain very different shares of different assets.
  The percent view is a conversion.
- **D60 Direction of the simulation.** Owner decision 2026-09-29: topics are
  built from prices (G.5.1); asset returns are not modified. The alternative
  (adding a topic component to real returns) keeps exact betas but changes
  the asset series; recorded as not chosen.
- **D61 Feasibility scaling** at `W_k R W_k' <= 0.95` (G.5.1), recorded and
  shown.
- **D62 Observed shocks** follow D9 on simulated attention levels; the
  estimators never see the designed shocks. The truth is defined on the
  observed standardised shocks (G.5.3), so it includes the attenuation from
  the trailing-mean construction.
- **D63 Truth with spillovers.** `B_true` is the population multivariate
  sensitivity and is not sparse. Recovery metrics use `B_true` for magnitudes and
  signs and the design links for coverage (G.8 point 5).
- **D64 Lead.** `l = 0` default (same-day, the plan's risk reading); `l = 1`
  for the predictive reading. The same `l` is used by the simulation and both
  estimators.
- **D65 Out-of-sample discipline.** Shock standardisation, return
  standardisation, estimated sensitivities and `Gamma` use the training window only.
- **D66 Direct estimator default**: elastic net with the universal penalty.
  Reason: it is the plan's baseline estimator, needs no cross-validation (fast
  enough to re-run on every control change), and adapts to 10 or 500 topics.
- **D67 Explained-variation metric**: uncentered OOS R2 over the forecast
  window, topics only (no intercept).
- **D68 Contribution**: frozen training sensitivities times realised window
  shocks (G.8 point 3), residual explicit; the in-window regression is
  rejected for the reasons in G.8. Since D76 the variance share is the
  default view and the return attribution the alternative.
- **D69 Correlation-table conventions**: blanks count as zero in the AVERAGE row;
  default blank rule is "not selected by the estimator". **Amended
  2026-09-30** (owner request): the blank rule follows the view. The true
  sensitivity blanks the pairs below the true-sensitivity threshold
  (`|B_true| < tau`, the run's `DirectConfig.select_tau`, in standardised
  units whatever the unit shown), the same test as "truly sensitive" in the
  MCC (G.8 point 5); the set sensitivity blanks the pairs with no link
  (`W = 0`); the estimated sensitivity, the OOS correlation and the OOS
  contribution keep "not selected by the estimator". Reason: with one rule
  for every view, the True view hid the pairs the estimator missed (211 of
  the 354 truly sensitive pairs at the defaults), and the Set view hid 45 of
  the 95 links and showed 159 selected pairs without a link as zeros. The
  rule stays on by default in every view (`_ui.blank_rule`,
  `_ui.blank_keep`).
- **D70 BKS in the lab**: weekly periods, half-life-matched `xi`,
  inverse-volatility asset weighting, 2% tolerance rule on a 12-point grid,
  train-until fit and in-window OOS factors; the per-topic split is labelled
  as not identified (D52).
- **D71 Caching and reproducibility**: stage keys from config hashes; one seed
  per random component (universe, links, noise), so changing one setting does
  not redraw unrelated components. The links use `ExposureConfig.seed` and
  the noise and attention components `ExposureConfig.noise_seed` (split on
  2026-09-29; before that one seed drove both).
- **D72 Dependencies**: optional extra `lab` (streamlit, plotly, yfinance,
  scikit-learn, pyarrow). The core estimator keeps its dependencies (D41).
- **D74 One oracle definition.** Set 2026-09-29 in the lab review, following
  G.8 as written. The oracle is the direct estimator with `b = B_true` and
  the estimator's training scales (`sd_train(r_n)` for returns,
  `sd_train(z_k)` for shocks); the code had used the full-sample volatility
  `asset_vol`, which includes the forecast window. Reasons: no
  forecast-window data enters the OOS benchmark, the `oracle` method
  reproduces the oracle line exactly (before, median 20.87% against 21.68%
  for 12 weeks from 2024-06-03, and up to 8.6 points on INR v USD), and the
  recovery metrics already compare `b` with `B_true` in the same units.
  `B_true` itself stays defined with full-sample standardisation; the two
  standardisations differ by a few percent per asset and topic. The `%` view
  converts every sensitivity metric with the training volatility.
- **D75 Reading of 'A v B'.** Owner decision 2026-09-29: a listed asset
  'A v B' is long leg A and short leg B; outrights and 'XXX v USD' pairs are
  long against cash. The owner's example table marks positions with 'L' and
  'S' (e.g. 'S Switzerland Large Cap v World EQ' is a short view of the
  Swiss-minus-World spread); the correlation table's per-asset long/short view
  reproduces that without changing the asset definition.
- **D76 Contribution views.** The variance share (G.8 point 4) is the default
  view of the contributions tab, because summed D9 shocks telescope and the
  return attribution understates the topics' role (G.8 point 3). The chart
  draws the residual and the realised move in a second panel with its own
  scale; return attribution is shown in percentage points, the variance
  share in percent. The alternative of attributing the cumulative change in
  attention levels was not chosen: it would no longer be the plan's
  attribution chain (frozen sensitivity times shock).
- **D77 Config normalisation.** Every lab config coerces its fields on
  construction (numbers to the annotated `int` or `float`, dates to ISO
  strings, lists to tuples), so a config read from JSON or YAML equals,
  hashes and keys like the one built in code. `scripts/run_lab.py` no longer
  patches configs.
- **D78 Truth with missing returns**: per observation mask, as in G.5.3.
- **D79 BKS OOS R2 disclosure and guards**: the shuffled-instrument
  reference, `K` below the number of assets, ridge 0 at `lambda = 0`, and the
  mid-week coverage notes of G.7.2. In the dashboard the BKS R2 no longer
  shares a tile with the direct R2, and the comparison always uses the
  current direct fit.
- **D80 Shared dashboard session.** One cache serves every browser session
  (G.10). A browser session reuses a cached BKS fit automatically only when
  it requested that fit itself. The BKS run makes no Streamlit call while it
  computes, so a widget change during a long fit no longer discards it.
- **D81 Training floor of one month.** Owner request 2026-09-29: the shortest
  training window is 21 weekdays (was 250). Measured on the default setup,
  scored over the 39 four-week windows of 2023–2025: median OOS R2 0.0% at one
  month, 6.2% at three months, 12.1% at one year and 15.1% at eight years,
  against an oracle of 15–19%. The lab's sensitivities are constant, so it shows
  only the noise cost of short windows; with drifting sensitivities in real data,
  shorter windows adapt faster. BKS keeps its own minimum of 26 weeks (24
  since D84).
- **D82 Real exposures page** (renamed "Real data" in D85). Owner request
  2026-09-29: a separate page at the top level, not a tab of the simulation,
  because it will show estimates on real news with no truth or oracle.
  Until the research pipeline delivers,
  it is a placeholder that states what it will show, the data contract (TBC)
  and which input files are present, and previews the sensitivity table once
  `data/real/sensitivities.parquet` exists (G.14; `exposures.parquet` until
  D89).
- **D83 Method comparison.** Owner request 2026-09-29: compare BKS with the
  direct estimator, and any method with the oracle. Every method delivers
  estimated sensitivities `B_hat` with the same observed shocks and training scales and is
  scored with the same forecast-window evaluation and window sweep (G.8), so
  nothing is fitted inside the forecast window. BKS-implied's instruments
  also weigh the history before the training window; the Compare tab says
  so (G.15). BKS enters through its
  implied sensitivities: the topic covariances its factor betas imply (BKS
  Eq. 5), turned into sensitivities with the training covariance of the topic
  shocks (G.15). Not chosen: BKS's own OOS R2, which fits `K` factors to
  each forecast week (D79). The sidebar's direct method keeps its settings;
  the other direct methods use the `DirectConfig` defaults with the same
  `tau`. A comparison never starts a BKS fit, and the dashboard reuses only
  a BKS fit its browser session requested (D80). On the dashboard defaults
  the median OOS R2 in the forecast window was 16.8% (elastic net), 26.5%
  (ridge), 8.2% (OLS), 7.0% (BKS-implied) and 23.2% (oracle). D88 adds the
  training-window variant.
- **D84 Time windows panel and dashboard defaults.** Owner request
  2026-09-29: the dashboard sets the training window by its end (cut-off)
  and a length of one month to 10 years counted back, with the start
  derived as `(cut-off + 1 day) - length`, extended to 21 weekdays when
  shorter and clipped at the data start (G.6). The alternative
  `cut-off - length + 1 day` gives 2024-12-31 for the defaults instead of
  2025-01-01; the two differ only for month-end cut-offs. Dashboard
  defaults: cut-off 2025-06-30, 6 months, forecast 2025-07-01 for 4 weeks.
  The library `WindowConfig` defaults are unchanged, so scripts and tests
  keep their results. The BKS minimum is 24 training weeks
  (`bks.MIN_TRAIN_PERIODS`), replacing the 26 of D81: a six-month window
  holds 25 or 26 week ends, so the default window runs BKS with a margin of
  one to two weeks.
- **D85 Real data page and shared settings panel.** Owner request
  2026-09-29: the Real exposures page is renamed "Real data" (title and URL
  path `real-data`; the module keeps the name `real_exposures.py`). The
  sidebar is shared by both pages, and the Real data page lists the settings
  in use (G.14). Invalid settings show there as a warning, not an error,
  because nothing runs on that page yet. Long-term intent: the shared panel
  will drive the real-data estimation, so method parameters and training
  windows can be tweaked and compared on real data as in the simulation.
- **D86 Ridge GCV counts the intercept.** Review of 2026-09-29: the GCV
  degrees of freedom are `df = 1 + sum_i d_i^2 / (d_i^2 + n lam)`, and a grid
  point with `df >= n` gets `GCV = inf`. Without the 1, a window with
  `L >= n_train - 1` (20 topics on the one-month minimum of D81)
  interpolates the centred returns as `lam -> 0`, GCV tends to 0 there, and
  the grid's lower edge won: 47 of 55 assets on a one-month window to
  2025-06-30, median OOS R2 -2,824% in the forecast window. With the 1:
  1.4% (one month to 2025-02-28) and -1.3% (to 2025-06-30); 26.5% on the
  six-month default and 16.1% on the eight-year library window, as before.
  `fit_direct` records `gcv_at_lower_edge`; the Compare table's ridge note
  and the Overview caption name it.
- **D87 Training checks near the data start.** Review of 2026-09-29: the
  21-day floor of D81 counts weekdays with a topic shock (a one-month window
  ending 2015-01-30 had 21 weekdays but 16 training pairs, and every asset
  was skipped), and the earliest cut-off is the 21st shock day (2015-02-06
  at `w = 5`). The BKS checks of the dashboard (the short-window note, the
  Compare tab and the BKS tab) count the weeks BKS can use after its 52-week
  burn-in (`bks.training_weeks`, exact against the built panel), so a
  window such as 2015-07-01 to 2015-12-31 is refused before a run with the
  reason, and the Run BKS buttons are disabled.
- **D88 Training-window BKS variant.** Owner request 2026-09-30: a BKS
  variant whose covariance instruments use the training window only, so that
  BKS sees exactly the data the direct methods see. `BKSLabConfig.history` is
  `"full"` (default, unchanged) or `"training"` (G.7.2): returns from
  `train_start`, attention from `w` weekdays before it, the training standard
  deviation as the inverse-volatility divisor, a burn-in of 0 weeks and 3
  days per instrument (`burn_in_weeks_training`, `min_days_training`). The
  comparison has both as separate methods, `bks_implied` ("BKS-implied (full
  history)") and `bks_implied_train` ("BKS-implied (training window)"), each
  with its own BKS fit of the sidebar's BKS model (`compare.method_config`);
  the new method is appended before the oracle, so the colour slots of the
  other methods stay. The sidebar's "Covariance history" radio drives the BKS
  tab and the sidebar's Run BKS; the Compare tab's Run BKS fits every
  selected variant not fitted yet. Not chosen: a trailing volatility inside
  the window (63 days of warm-up, about half of a six-month window), and the
  kernel truncation `CovarianceConfig.lookback_periods` (the earlier training
  weeks and the divisor would still read days before the window). Measured
  2026-09-30, median OOS R2 in the forecast window, noise seed 0 (the
  dashboard's): on the dashboard defaults -16.2% for the training variant
  against 7.0% for the full history (24 against 26 training weeks; lambda
  0.017 with all 20 topics kept against 0.277 with 10; Spearman with `B_true`
  0.11 against 0.19); on the library window (2015–2022) -0.1% against -2.1%
  (415 against 352 training weeks, as the training variant has no burn-in;
  Spearman 0.32 against 0.16). Seed 0 is one draw and the best of five for
  both variants on the defaults: over noise seeds 0–4 the training variant
  gives -16% to -107% and the full history -18% to 7%, the training variant
  lower in every seed. On the library window neither is ahead in every seed
  (means over seeds 0–4: -1.4% and -0.1%). The variants also differ in the
  return divisor, the burn-in, the day minimum and the number of training
  weeks, so the gap is not all history (G.15). Plain
  IPCA (`K = L`, `lambda = 0`) on the training window matches OLS to 0.07
  (lead 0) and 0.10 (lead 1) on a six-month window and 0.03 on eight years.
- **D89 Terminology: topic sensitivity.** Owner decision 2026-09-30: the
  topic-to-asset quantity `b_{k,n}` (the matrix `B`) is called **topic
  sensitivity** (G.0) in the documents and the dashboard text, not
  "exposure". Reason: "exposure" is easily read as a dollar exposure to the
  asset. The values set with the link map are set sensitivities (`W`, the
  "betas"), `B_true` is the true and `B_hat` the estimated sensitivity. The
  real-data contract becomes `sensitivities.parquet` with a `sensitivity`
  column (G.14). The dashboard's Correlation table tab keeps its name. The
  code keeps the older word: "exposure" in code identifiers (`exposure_lab`,
  `ExposureConfig`, `exposure_heatmap`, `B_hat`, and the "exposures" in
  docstrings and variable names) means topic sensitivity. Earlier names of
  pages and tabs ("Real exposures", "Exposure table") stay in the entries
  that record them.
- **D90 BKS trace page.** Owner request 2026-09-30: the BKS implementation
  "does not behave as expected"; a page that traces what happens in which
  step, with the inputs, the intermediate results and the outputs, and shows
  what each result should be, without changing values. A third page in the
  top navigation, **BKS trace** (URL path `bks-trace`, between Simulation lab
  and Real data), linked from the BKS tab (under Run BKS) and from the Compare
  methods tab's inspect section (for a BKS-implied variant), with a link
  back. It traces the BKS run of the current settings and the sidebar's
  covariance history, the same run the BKS tab shows: Run BKS works on both
  pages, both handle its requests (`_bks_sync`) and keep the same result, and
  the trace needs a fit this browser session requested (D80). The trace is
  the library module `exposure_lab/trace.py` (pure, cached as the session
  stage `bks_trace`, G.10); the page is `dashboard/trace_page.py`. Every step
  shows its result next to an independent reference, a recomputation by the
  formula, an identity or the simulation's population value, and the Summary
  step ranks the losses along a ladder of variants from the true sensitivities
  to the production BKS-implied ones (G.16). Reason for a separate page, not
  a tab: Streamlit runs the code of every tab on every rerun, so a trace tab
  would slow every widget change on the simulation page, and a page has its
  own address and loads only when opened. Not chosen: editing intermediate
  values (the page reads, the owner changes settings in the sidebar);
  keeping the daily shocks and the covariance array in the cached panel (1.1
  GB at 500 x 500; the trace recomputes the selected asset's instruments).
  Findings of the first trace, 2026-09-30, dashboard defaults: every identity
  check holds (timing, lag, units, instruments, panel, fit, forecast factors
  and the Eq. 5 chain reproduce to about 1e-12 or better); the departures are
  the Eq. 5 projection (the fit's K directions keep 38% of the instruments'
  squared norm against 91% for the best three; Spearman with `B_true` 0.19
  against 0.71 for the instruments alone), the moment windows of the full
  history (instruments 92% before the training start, `Sigma_z` over the 129
  training days; 0.91 with `Sigma_z` over the instruments' own history) and a
  lambda choice within the noise of the in-sample Sharpe ratio (G.16).

### G.12 Limitations of the lab

1. **Simulated attention.** Topic attention is built from the returns it is
   meant to explain, so the lab shows estimator behaviour and the size of
   explainable variation under stated assumptions, not the information
   content of real news.
2. **Illustrative link map.** The default links and signs are a plausible
   story, not estimates; attention is unsigned in reality, and the sign
   convention is a modelling assumption per topic.
3. **The truth is a full-sample summary.** The population truth uses the
   full-sample moments of the filtered noise-free signal, so it keeps the
   serial correlation of daily returns (a Monte Carlo on the real data
   matches it to sampling error). It is exact for a stationary process with
   these moments; it does not model volatility clustering or correlations
   that change over time, and it ignores the clipping of attention at `1e-6`.
4. **Non-synchronous closes.** Legs close in New York, London, Frankfurt,
   Zurich and Tokyo; daily spread returns mix closing times. Weekly BKS
   periods reduce this.
5. **Spot FX and bucket mismatches.** FX legs exclude carry; UK, Italy and
   Japan 7–10y legs use all-maturity government bond funds (TBC in
   `data/market/README.md`).
6. **Short windows.** One-week windows have five daily observations; OOS
   correlations and R2 over them are dominated by noise. The window sweep
   shows the distribution across windows.
7. **BKS identification.** D47 and D52 apply unchanged: noise topics'
   instruments inherit betas, and the per-topic split of BKS fitted returns
   is not identified. The BKS OOS R2 fits each week's factors to that week's
   returns and is not comparable to the direct R2 (D79).
8. **BKS-implied sensitivities.** The method comparison scores BKS through the
   sensitivities its factor betas imply (G.15). They depend on which topics the
   sparse fit kept (D52), and they keep only the part of each asset's topic
   covariances that lies in the `K` directions BKS fitted to price weekly
   returns. On the lab data those directions carry little of the topic
   signal, so BKS-implied scores below the direct methods (G.15.1).
   With the full history their instruments also weigh the days before the
   training window (about 92% of the kernel weight on the dashboard
   defaults), so that variant is not strictly like for like. The
   training-window variant (D88) is, at the cost of noisy instruments in its
   first training weeks.

### G.13 Module contracts

All in `narrative_ipca/exposure_lab/`; configs in `config.py`, containers in
`types.py`. Topic x asset matrices have topics as rows; the asset x topic frames of
`WindowEval` (the correlation-table layout) have assets as rows.

```python
# reference.py — data/reference and data/market locations and loaders
def data_dir() -> Path                     # repo data/; env NARRATIVE_IPCA_DATA_DIR overrides
def load_assets() -> pd.DataFrame          # index asset_id in image order: order, name, long_leg, short_leg, asset_class, sub_class
def load_legs() -> pd.DataFrame            # index leg_id
def load_topics() -> pd.DataFrame          # index topic_id: order, ontology, group, name, scope
def load_default_links() -> pd.DataFrame   # topic_id, asset_id, tier, sign, mechanism (validated)

# market.py — G.2
def weekday_calendar(start: str, end: str) -> pd.DatetimeIndex
def generic_asset_table(n: int, seed: int) -> pd.DataFrame
def artificial_returns(assets: pd.DataFrame, calendar: pd.DatetimeIndex, seed: int) -> pd.DataFrame
def load_real_returns(asset_ids: list[str], start: str, end: str) -> tuple[pd.DataFrame, dict]
def build_market(cfg: UniverseConfig) -> MarketData

# links.py — G.3, G.4
def build_topic_table(cfg: TopicSetConfig) -> TopicTable
def build_link_map(topics: TopicTable, assets: pd.DataFrame, topic_cfg: TopicSetConfig,
                   exposure_cfg: ExposureConfig) -> LinkMap
def design_matrix(links: LinkMap, topics: TopicTable, assets: pd.DataFrame,
                  exposure_cfg: ExposureConfig) -> pd.DataFrame          # W_unscaled

# dgp.py — G.5
def simulate_lab(cfg: LabConfig, market: MarketData | None = None) -> SimData
def truth_for_window(sim: SimData, shock_window: int) -> SimTruth
def observed_shocks(attention: pd.DataFrame, shock_window: int, train_start: str,
                    train_end: str) -> ObservedShocks
def observation_groups(obs: np.ndarray) -> list[tuple[np.ndarray, list[int]]]   # D78

# direct.py — G.7.1
def fit_direct(sim: SimData, shocks: ObservedShocks, cfg: DirectConfig,
               truth: SimTruth) -> DirectFit

# evaluate.py — G.8
def window_return_days(calendar: pd.DatetimeIndex, window: WindowConfig) -> pd.DatetimeIndex
def evaluate_window(sim: SimData, shocks: ObservedShocks, fit: DirectFit,
                    window: WindowConfig, truth: SimTruth) -> WindowEval
def recovery_metrics(fit: DirectFit, truth: SimTruth, tau: float) -> dict[str, float]
def window_sweep(sim: SimData, shocks: ObservedShocks, fit: DirectFit, window: WindowConfig,
                 truth: SimTruth, max_windows: int = 150) -> pd.DataFrame
def median_finite(values) -> float

# bks.py — G.7.2, G.15
MIN_TRAIN_PERIODS = 24                     # fewest training weeks fit_bks accepts (D84)
IMPLIED_METHOD = "bks_implied"             # IMPLIED_NOTE: the D52 caveat of the implied sensitivities
IMPLIED_TRAIN_METHOD = "bks_implied_train" # the training-window variant (D88)
IMPLIED_METHODS = {"full": IMPLIED_METHOD, "training": IMPLIED_TRAIN_METHOD}; HISTORY_LABELS
def bks_pipeline_config(cfg: BKSLabConfig, shock_window: int, lead_days: int,
                        n_assets: int | None = None) -> PipelineConfig   # burn-in and min_days of cfg.history
def training_weeks(train_start, train_end, cfg: BKSLabConfig, lead_days: int = 0, calendar=None,
                   shock_window: int = 5) -> tuple[int, pd.Timestamp]   # usable weeks, first usable week (D87, D88)
def kernel_history_share(train_start, train_end, cfg: BKSLabConfig, lead_days: int = 0, calendar=None,
                         shock_window: int = 5) -> float  # kernel weight before train_start (G.15); 0 for "training"
def build_bks_panel(sim: SimData, cfg: BKSLabConfig, shock_window: int, *, train_start=None,
                    train_end=None) -> BKSPanel             # the window is needed for cfg.history = "training" (D88)
def fit_bks(panel: BKSPanel, cfg: BKSLabConfig, train_end: str, progress=None, *,
            train_start: str | None = None) -> BKSFit
def evaluate_bks(panel: BKSPanel, fit: BKSFit, window: WindowConfig) -> BKSLabResult
def implied_topic_covariance(C: np.ndarray, Gamma: np.ndarray, rcond: float
                             ) -> tuple[np.ndarray, np.ndarray, int]           # (m, beta, rank of Gamma_tilde)
def implied_exposures(panel: BKSPanel, fit: BKSFit, sim: SimData, shocks: ObservedShocks,
                      select_tau: float = 0.05) -> DirectFit   # method "bks_implied" or "bks_implied_train" by history

# compare.py — G.15
METHODS = ("elastic_net", "ridge", "ols", "bks_implied", "bks_implied_train", "oracle")  # display and colour order
DIRECT_METHODS, BKS_METHODS, BKS_HISTORY, METHOD_LABELS, SUMMARY_COLUMNS, ORACLE_NOTE
def method_config(cfg: LabConfig, method: str) -> LabConfig   # the sidebar's method as is, others DirectConfig
                                                              # defaults; BKS-implied: cfg.bks with its history
def method_label(method: str, fit: DirectFit | None = None) -> str
@dataclass
class ComparisonResult:      # summary (method x SUMMARY_COLUMNS), r2 (assets x available methods),
    ...                      # r2_sweep (start, end, method, median_r2), fits, evals, meta
def compare_methods(sim: SimData, shocks: ObservedShocks, truth: SimTruth, window: WindowConfig,
                    fits: dict[str, DirectFit], fit_seconds: dict[str, float] | None = None,
                    unavailable: dict[str, str] | None = None) -> ComparisonResult

# session.py — cached orchestration used by the dashboard and scripts/run_lab.py
class LabSession:            # stage results memoised by LabConfig.key(stage); LabSession(max_entries=6, bks_max_entries=2)
    def market(cfg) / simulation(cfg) / truth(cfg) / shocks(cfg) / direct(cfg) / evaluation(cfg) / sweep(cfg) / bks(cfg)
    def bks_implied(cfg) -> DirectFit                   # from the cached BKS panel and fit; LookupError if not cached
    def bks_trace(cfg) -> BKSTrace                      # G.16: from the cached BKS panel and fit; LookupError if not cached
    def method_fit(cfg, method: str) -> DirectFit
    def comparison(cfg, methods=None, use_bks: bool | Iterable[str] = True,
                   bks_errors: Mapping[str, str] | None = None) -> ComparisonResult
                                                        # never starts a BKS fit; use_bks may name the variants allowed;
                                                        # bks_errors: a refused fit's reason (BKS_REFUSED prefix)
    def stage_key(stage, cfg, *, methods=None, bks_token: str | dict = "nobks") -> str
    def comparison_key(cfg, methods=None, use_bks=True) -> str
    def bks_ready(cfg, method=None) -> bool
BKS_NOT_RUN, BKS_OFF, BKS_REFUSED                       # reasons shown for an unavailable BKS-implied fit
def run_lab(cfg: LabConfig, with_bks: bool = False, progress=None, with_compare: bool = False) -> dict[str, Any]
                             # with both flags also fits the other BKS variant as stage bks_fit_training
                             # (or bks_fit_full), timed and keyed; when it refuses, the comparison lists it
                             # as unavailable with the reason (review 2026-09-30)

# charts.py — Plotly figure builders, pure, no Streamlit
def exposure_heatmap(values, *, blank, value_label, row_prefix=None, average_row=True, max_cols=40,
                     colorscale="diverging", row_title="Asset", col_title="Topics") -> go.Figure
def contribution_bars(contrib, *, true_contrib=None, realized, residual, top_n=15, units="pp",
                      realized_label="Realised move", residual_label="Not explained by topics",
                      axis_title=None) -> go.Figure
def r2_bars(r2, r2_oracle, r2_true=None, *, labels=None, title=None, clip=-1.0,
            name="Estimator") -> go.Figure
def cumulative_explained(realized_daily, fitted, fitted_oracle) -> go.Figure
def window_sweep_chart(sweep, *, empty_message="No forecast windows to show") -> go.Figure
def attention_chart(levels, shocks, window) -> go.Figure
def gamma_norm_bars(norms, selected) -> go.Figure
def lambda_path_chart(path, lam_star) -> go.Figure
def exposure_scatter(estimate, truth, *, linked=None, title=None) -> go.Figure
def method_styles(methods, reference="oracle", slots=None) -> dict[str, dict]   # colour, symbol, dash; slots fix them
def method_r2_dots(r2, *, labels=None, method_labels=None, reference="oracle", title=None,
                   clip=-1.0, slots=None) -> go.Figure                   # assets x methods, sorted by the oracle
def method_sweep_lines(r2_sweep_long, *, method_labels=None, reference="oracle", title=None,
                       empty_message="No forecast windows to show", clip=-1.0,
                       slots=None) -> go.Figure
# BKS trace page (G.16, D90)
def line_panels(panels, *, shade=(), markers=(), x_title=None, x_log=False, title=None, subtitle=None,
                height=None) -> go.Figure          # 1-3 stacked panels of named series on one date or numeric axis;
                                                    # shaded windows (start, end, label), dashed markers (x, label)
def matrix_heatmap(values, *, row_labels=None, col_labels=None, value_label="value", zmax=None, show_text=None,
                   max_rows=80, max_cols=45, title=None, subtitle=None, row_title=None, col_title=None,
                   highlight_rows=(), decimals=None) -> go.Figure   # signed matrix around 0; truncation noted
def ladder_chart(ladder, *, metrics=(("spearman", ...), ("median_r2", ...)), label_col="label", highlight=None,
                 reference=None, title=None, subtitle=None) -> go.Figure   # variants top to bottom, a bar panel per metric
def identity_scatter(x, y, *, labels=None, highlight=None, x_title, y_title, title=None, subtitle=None,
                     fit_line=True, identity_line=True, highlight_label="Highlighted", point_label="Points",
                     max_points=20000) -> go.Figure   # 45-degree line; least-squares slope and correlation over all points
def grouped_bars(frame, *, labels=None, series_labels=None, colors=None, axis_title=None, reference=None,
                 orientation="h", top_n=None, sort_by=None, title=None, subtitle=None, percent=False,
                 separate=False) -> go.Figure       # categories x series; at most 120 rows unless top_n
def lambda_trace_chart(path, *, lam_star=None, lam_best=None, band_floor=None, criterion_label=...,
                       extra=None, extra_labels=None, extra_title=None, title=None, subtitle=None,
                       null_band=None, band_floor_alt=None) -> go.Figure
                                                    # criterion with a +-1 se band, the tolerance floor, the null
                                                    # band of a Sharpe with no priced signal; selected topics; recovery
def coefficient_path_chart(norms, *, selected=(), labels=None, lam_star=None, max_colored=8, title=None,
                           subtitle=None) -> go.Figure   # standardised Gamma row norms along lambda (log axis)

# trace.py — G.16: a cached lab BKS run traced step by step, with references (pure, no Streamlit)
STEPS: dict[str, str]                               # summary, inputs, align, shocks, instruments, panel, fit, forecast, implied
LADDER: dict[str, str]                              # reference ladder variants, top to bottom (G.16)
@dataclass class TraceCheck: step, name, relation, value, reference, tolerance, status ("ok" | "off" | "info"),
                             note, kind ("identity" | "diagnostic")
@dataclass class BKSTrace:                          # recomputed daily shocks, population references, the Eq. 5 chain
    ...                                             # for every asset, ladder, capture shares, path and KKT tables,
    def checks_frame(step=None) -> pd.DataFrame     # step tables, checks, findings, meta (settings, shapes, timings)
    def status_frame() -> pd.DataFrame              # one row per step: checks ok, key number, reading
def build_trace(panel: BKSPanel, fit: BKSFit, result: BKSLabResult, sim: SimData, shocks: ObservedShocks,
                truth: SimTruth, window: WindowConfig, bks_cfg: BKSLabConfig, implied: DirectFit,
                *, path_trace: bool | None = None) -> BKSTrace   # path_trace None: on up to 60 topics
def population_moments(sim, truth) -> dict          # var_z, sd_z, S_z, Cov(z, r), C (the truth's internals)
def window_truth(sim, shocks, truth) -> pd.DataFrame    # the population value on the training window's own signal
# per-selection detail (not cached; one asset, topic or week)
def asset_days(trace, panel, sim, asset) / topic_days(trace, panel, sim, shocks, topic) -> pd.DataFrame
def instrument_series(trace, panel, asset, topic) -> pd.DataFrame      # every instrument week, panel value, reference
def kernel_profile(trace, panel, asset, topic, return_week) -> tuple[pd.DataFrame, dict]   # weights, contributions
def instrument_row(trace, panel, sim, asset, return_week) -> pd.DataFrame   # panel, brute force, signal, noise
def pairing_table(trace, panel, sim, asset) / design_matrix(trace, panel, fit, return_week) -> pd.DataFrame
def forecast_week(trace, panel, fit, result, week) -> tuple[pd.DataFrame, dict]
def asset_chain(trace, asset, direct=None) -> pd.DataFrame             # the Eq. 5 chain of one asset, per topic
def lambda_path_trace(panel, fit, sim, shocks, truth, window, bks_cfg) -> pd.DataFrame   # per-lambda refit
```

Dashboard helpers (`dashboard/_ui.py`, pure, no Streamlit) added on
2026-09-29:

```python
def training_window(train_end, months, min_days=None) -> dict   # start, end, n_days, corrections, caption (G.6)
def first_shock_day(shock_window) -> date / shock_days(train_start, train_end, shock_window) -> int
def earliest_train_end(shock_window, min_days=None) -> date        # the cut-off control's minimum (D87)
def bks_training_check(train_start, train_end, bks_cfg=None, lead_days=0,
                       shock_window=5) -> dict                  # weeks, n_weeks, can_run, reason (both histories)
def plural(n, word, words=None) -> str
def settings_in_use(values, asset_classes=None, asset_names=None) -> pd.DataFrame   # Group, Setting, Value (G.14)
def comparison_table(summary, notes=None) -> pd.DataFrame        # the Compare methods table, shares in percent
def method_option_label(method, direct=None) -> str
def unavailable_note(note) -> str                                 # a method's reason in plain words
```

### G.14 Real data page (placeholder, D82, D85)

`dashboard/real_exposures.py`, the second page of the top navigation, titled
"Real data" at the URL path `real-data` (named "Real exposures" until
2026-09-29; the module keeps its name). Nothing on it is simulated.

**Settings in use** (D85). The sidebar is shared with the simulation lab.
Near the top of the page, after the placeholder note and before the file
status, a table (Group, Setting, Value) lists the settings that will apply
to real data:

1. **Time windows**: cut-off, training length, the resulting training window
   (with any correction of G.6), forecast start and length, shock window.
2. **Direct estimator**: method and its penalty settings, selection
   threshold `tau`; the oracle is marked as not available on real data.
3. **BKS model**: `K`, kernel half-life with the weekly `xi`, lambda rule,
   grid, intercept penalty, covariance history (D88).
4. **Asset selection**: listed assets kept, asset classes, left-out assets.

A line under the table names the simulation-only settings that do not apply
(price source, generic assets and topics, betas and link seeds). Invalid
settings show as one warning, not an error. Long-term intent (owner,
2026-09-29): the shared panel will drive the real-data estimation, so method
parameters and training windows can be tweaked and compared on real data as
in the simulation lab.

**What it will show:**

1. **Sensitivity table**: the estimated sensitivity of each asset to each topic
   (% return per one-sd attention shock) with its uncertainty, in the layout
   of the simulation's correlation table, for a chosen estimation date.
2. **Topic contributions**: for a chosen asset and period, the frozen
   sensitivities times the realised attention shocks (G.8 point 3), ranked with
   the largest on top, plus the part not explained by topics.
3. **Explained variation over time**: out-of-sample R2 per asset on
   consecutive windows (G.8 point 6).
4. **Provenance**: each sensitivity's source, estimation window and coverage.

**What it needs (data contract, TBC)**, in `data/real/`:

| File | Content | Columns |
|---|---|---|
| `topics.csv` | one row per topic | `topic_id`, `name`, `taxonomy_class`, `origin` (fastopic or manual), `model_version` |
| `attention.parquet` | daily attention level per topic | date index; one column per `topic_id` |
| `sensitivities.parquet` | one row per (`as_of`, `topic_id`, `asset_id`) | `as_of`, `topic_id`, `asset_id`, `sensitivity`, `uncertainty`, `source` (structural, regression, llm or blended), `effective_window`, `coverage` (full, partial or none), `version` |

Assets and returns come from `data/reference/` and `data/market/`. The column
names follow the interface fields of the research plan. Today the page shows
the status of each file and, when `sensitivities.parquet` exists and passes its
checks (required columns, allowed `source` and `coverage` values, no
duplicate keys), the sensitivity table of the chosen date with `coverage = none`
blank. Views 2 to 4 are built when the data arrives.

**How it differs from the simulation lab:** there is no truth and no oracle,
so sensitivities are judged on realised returns and on the golden set; the
sensitivities come from the plan's estimation (`B = M·L + S`), not from the lab's
direct regression; shocks and contributions use the lab's definitions, so the
views are comparable.

**Open points (TBC):** plain or versioned topic ids (`T017` or `T017@v3`);
whether `sensitivity` is the composed `B` only or the factor route and the
overlay are delivered separately; whether the lead (same day or next day)
and the manual topic set belong in the Settings in use list (today they are
neither listed nor named as simulation-only).

### G.15 Method comparison (D83)

The Compare methods tab answers two owner questions (2026-09-29): how does
BKS compare with the direct estimator, and how does a chosen method compare
with the oracle? Code: `compare.py`, `bks.implied_exposures`, and the
`bks_implied` and `comparison` stages of `session.py`.

**Like-for-like design.** Every method delivers the same object, a
`DirectFit`: estimated sensitivities `B_hat` (topics x assets, standardised units) with the
same observed shocks, returns standardised by their training standard
deviation and shocks by the training standard deviation of `z`. The direct
methods are fitted on the training pairs only. The forecast-window
evaluation (G.8 points 1–5) and the window sweep (G.8 point 6) then apply
unchanged:

1. Nothing is fitted inside the forecast window; every method's sensitivities
   are frozen at the training end (D65).
2. Every method is scored on the same forecast return days against the same
   oracle line: the oracle uses each fit's training scales (D74), and every
   method computes them on the same training pairs.
3. The methods are `elastic_net`, `ridge`, `ols` (G.7.1), `bks_implied`
   and `bks_implied_train` (below) and `oracle` (`b = B_true`, the
   reference, not an estimator). The sidebar's direct method keeps its
   settings; the other direct methods use the `DirectConfig` defaults
   (universal penalty, ridge lambda by GCV) with the same selection threshold
   `tau`.
4. No return or attention after the cut-off changes any fit: tested by
   perturbation for the direct methods (`tests/test_lab_direct.py`) and for
   both BKS variants and their implied sensitivities at lead 0 and lead 1 with a
   Wednesday cut-off (`tests/test_lab_compare.py`). For the training-window
   variant, no return or attention before `train_start - w` weekdays changes
   its panel, `Gamma`, implied sensitivities or scales either (tested the same
   way).

**Two BKS variants (D88).** BKS enters the comparison twice, with the
sidebar's BKS model and two covariance histories (`BKSLabConfig.history`,
G.7.2):

1. `bks_implied`, "BKS-implied (full history)": the panel starts at the
   start of the data, as in BKS.
2. `bks_implied_train`, "BKS-implied (training window)": the panel starts at
   the training start, so BKS sees the data the direct methods see. It is
   the like-for-like BKS figure.

**A key limitation of the full-history variant: it sees more history than
the direct methods.** Its `Gamma`, `Sigma_z` and the scales use the training
window, but:

1. Its instruments are kernel covariances that weigh the whole history
   before the cut-off (half-life 69 months). On the dashboard defaults about
   92% of that kernel weight lies before the training start
   (`bks.kernel_history_share`, recorded as
   `meta["kernel_share_before_train"]`). On the library window (2015–2022)
   the kernel lies inside the training window.
2. The first training week's return can include days before the training
   start (2024-12-30 and 2024-12-31 on the defaults).
3. The extra history carries information: the instruments alone, through
   the same unit conversion, reach Spearman 0.71 with `B_true` and a median
   OOS R2 of 11.2% on the six-month window, against 0.55 and 16.8% for the
   elastic net (review measurement). The bias favours BKS; in the windows
   tested (6 and 24 months) it did not change the ranking.
4. The Compare tab states the share for the current window. The
   training-window variant has none of this history (share 0). On the
   dashboard defaults it scores below the full-history variant in each of
   the noise seeds 0–4 (-16.2% against 7.0% on seed 0, figures below). The
   variants also differ in the return divisor, the burn-in, the day minimum
   and the number of training weeks (G.7.2), so not all of the gap is the
   extra history: on a 10-year window to 2025-06-30, where only 1% of the
   kernel weight lies before the training start, the full history scored
   0.5% against 10.6% for the training window (seed 0, one forecast window).

**BKS-implied sensitivities.** The BKS OOS R2 fits `K` factors to each forecast
week's own returns (D79), so it cannot be set next to the direct R2. The lab
instead turns the BKS training fit into topic sensitivities and scores them like
any other method. Symbols, for asset `i` and `L` topics:

- `c_i = [1, cov_i]`: the asset's instrument row in the last training week:
  the constant, then its kernel covariances with the `L` topic shocks,
  measured at the end of the week before (the asset's latest training row
  when it has none that week).
- `Gamma`: the `((L+1) x K)` training map from instruments to factor
  loadings, with `K` the number of factors. `Gamma_0` is its first row (the
  constant) and `Gamma_tilde` its `L` topic rows; topics the group lasso
  dropped have zero rows.
- `beta_i = c_i Gamma`: the asset's `K` factor betas, the part BKS
  identifies (D52).

Steps:

1. **Model-implied topic covariances.**
   `m_i = Gamma_tilde (Gamma_tilde' Gamma_tilde)^+ beta_i'`, an `L`-vector,
   with `^+` the pseudo-inverse at the cut-off `EvaluationConfig.rcond`. In
   words: the covariances of the asset's return with the raw topic shocks
   that its factor betas imply, a rank-`K` reconstruction that is zero for
   dropped topics.
2. **Units.** Under `asset_weighting = "inverse_vol"` (the lab default) the
   panel's daily returns were divided by a divisor, so `m_i` is multiplied by
   the asset's mean divisor over its training return days. Full history: the
   divisor is a trailing volatility, and the conversion is approximate
   because it varies over time. Training window only: the divisor is the
   asset's training standard deviation on every day, and the conversion is
   exact. There is no conversion under `"none"`.
3. **Raw sensitivities.** `b_i = Sigma_z^+ m_i`, with `Sigma_z` the covariance of
   the raw observed shocks `z` over the direct estimator's training shock
   days. In words: the regression coefficients that the implied covariances
   give, as the population truth `B_true = S_z^{-1} C` does (G.5.3).
4. **Standardisation.** `B_hat[k, i] = b_i[k] * sd_train(z_k) / sd_train(r_i)`,
   with `sd_train(z_k)` the training standard deviation of topic `k`'s shock
   and `sd_train(r_i)` that of asset `i`'s return, computed exactly as
   `fit_direct` does. A pair is selected when `|B_hat| >= tau`.

**Why this formula.** BKS Eq. 5 writes the instruments as the model-implied
covariance of returns with topic attention, `cov_i = beta_i Sigma_ff A'`,
with `Sigma_ff` the factor covariance and
`A = Gamma_tilde (Gamma_tilde' Gamma_tilde)^-1 Sigma_ff^-1` the map from
factors to topic shocks that `wrapup.recover_A` estimates. Then
`A Sigma_ff beta_i' = Gamma_tilde (Gamma_tilde' Gamma_tilde)^-1 beta_i'`:
`Sigma_ff` cancels, and the implied covariance needs only `Gamma` and the
betas. Checks in `tests/test_lab_compare.py`:

1. `beta_i` equals `sparse_ipca.betas` exactly, and with the fit's factors
   reproduces its in-sample total R2 (so `Gamma` is in the units of the
   panel's instruments).
2. With `K = L` and `lambda = 0`, `m_i = cov_i' + (Gamma_tilde')^-1 Gamma_0'`
   to about 1e-14 relative.
3. Without the constant's term, the `"none"` and `"inverse_vol"` panels agree
   (maximum difference 0.026) and each matches OLS within 0.03.
4. Training-window variant with `K = L = 8` and `lambda = 0`: without the
   constant's term the implied sensitivities match OLS on the same window to
   0.074 (lead 0) and 0.104 (lead 1) on six months (sensitivities up to 0.59) and
   to 0.03 on eight years. They differ because the last instrument ends a
   week and a day before the cut-off and because of the kernel weights
   (`xi^26 = 0.94` over six months). Its last training instrument equals the
   kernel covariance of the training window's scaled returns and `z`,
   computed by hand.

**The summary table**, one row per method, the oracle last:

1. **Selected pairs**: topic-asset pairs with `b != 0` (sparse methods) or
   `|b| >= tau` (dense methods and BKS-implied).
2. **Coverage**: share of design-linked pairs selected.
3. **Sign agreement**: share of linked and selected pairs whose estimated
   sign equals the true sign.
4. **MCC**: Matthews correlation of "selected" against "truly sensitive"
   (`|B_true| >= tau`), over all pairs.
5. **Spearman vs truth** and **RMSE vs truth**: rank correlation and root
   mean squared difference of `B_hat` and `B_true`, over all pairs. Points
   1–5 are the recovery metrics of G.8 point 5.
6. **Median OOS R2, this window**: the median over assets of the per-asset
   OOS R2 in the forecast window (G.8 point 1).
7. **Median OOS R2, all windows**: the median over the sweep windows of the
   cross-asset median OOS R2 (G.8 point 6).
8. **Windows above the oracle**: the share of sweep windows where the
   method's cross-asset median R2 exceeds the oracle's. The oracle is not
   the best fit in every window, so a method can beat it by chance.
9. **Fit time**: seconds to fit on the training window; for BKS-implied, the
   BKS panel and fit plus the conversion.
10. **Note**: the reason when a method is not available (BKS not run on the
    current settings, OLS refused with `L >= n_train / 2`), the D52 caveat
    for both BKS-implied rows, "reference" for the oracle, and for ridge the number of
    assets whose GCV lambda is at the grid's lower edge (D86).

The dashboard shows points 6–8 right after the method name, then points
1–5, the fit time and the note, so the out-of-sample columns stay on screen
at laptop width.

**Measured on the dashboard defaults** (2026-09-29; 55 listed assets with
real prices, 20 manual topics, training 2025-01-01 to 2025-06-30, forecast
2025-07-01 for 4 weeks, `K = 3`). BKS with the full history ran on 26
training weeks (lambda 0.277, 10 topics kept). Median OOS R2 in the forecast
window: elastic net 16.8%, ridge 26.5%, OLS 8.2%, BKS-implied (full history)
7.0%, oracle 23.2%. Ridge is above the oracle in this window but in only 2 of
the 6 sweep windows (median over the sweep 8.3% against 10.5%); BKS-implied
(full history) is below the oracle in all 6 (median −8.5%). On the library
default window (training 2015–2022, 39 sweep windows), BKS-implied (full
history) reached −2.1% in the first window (Spearman with `B_true` 0.16)
against 16.8% for the elastic net and 20.3% for the oracle.

The training-window variant (2026-09-30): on the dashboard defaults it ran on
24 training weeks (lambda 0.017, all 20 topics kept) and reached −16.2%
(median over the 6 sweep windows −51.1%, Spearman 0.11); on the library
window it ran on 415 weeks (lambda 0.046, 18 topics kept) and reached −0.1%
(median over the 39 sweep windows 4.4% against −0.8% for the full history,
Spearman 0.32). These are noise seed 0; D88 gives the range over seeds 0–4.
Its tuned `K = 3` fit loses the signal in the same step as the full
history's (G.15.1).

**A key limitation (D52).** BKS identifies the assets' factor betas, not how
they split across topics, so the implied sensitivities depend on which topics
the sparse fit kept. They are a rank-`K` reconstruction: they keep only the
part of each asset's topic covariances that lies in the `K` directions of
`Gamma_tilde`, which the fit chooses to price weekly returns. On 30 generic
assets with 8 linked topics and `K = 3`, their Spearman correlation with
`B_true` was 0.40 to 0.49 and their sign agreement on the linked pairs 0.52
to 0.69 over three seeds. On the lab data the loss comes from which `K`
directions the fit keeps, not from having only `K` of them, and not from
topic selection or the data BKS sees (G.15.1). The table's note says so
(`bks.IMPLIED_NOTE`). When `Gamma_tilde` has rank
below `K` (on 30 generic assets with `K = 3` the tuned fit's third singular
value was 2e-6), the pseudo-inverse cut-off drops that direction and the
inspect caption names the number of directions used. A further
approximation: when the fit keeps the constant instrument, its implied
covariance is added to every asset (`meta["B_const"]`). The kernel history
before the training window is the full-history variant's limitation stated
under the like-for-like design above.

#### G.15.1 Why BKS-implied explains less than the direct methods

Owner question 2026-09-30, measured the same day. The main reason is the
step that turns the BKS fit back into sensitivities (step 1 above, BKS Eq. 5).
It keeps only the part of each asset's topic covariances that lies in the
`K` directions of `Gamma_tilde`, and IPCA chooses those directions to price
weekly returns, not to hold the topic signal. On the lab data they hold
little of it. Having only `K` directions is not the problem, and neither
are the history, the units or, at the tuned `K = 3`, the constant term.

**Set-up.**

- Setting A: the dashboard defaults (training 2025-01-01 to 2025-06-30, 6
  sweep windows). Setting B: the library window (training 2015–2022, 39
  sweep windows).
- Every figure is the mean over noise seeds 0–2, scored with
  `evaluate_window` and `window_sweep` as in the Compare tab; seed 0
  reproduces the dashboard. BKS rows use the full history unless stated.
- "Instruments only" is the implied-sensitivity formula with `m_i` replaced by
  the asset's own instrument covariances `cov_i` (steps 2–4 unchanged).
- Three identity checks hold to about 1e-15: OLS equals `Sigma_z^+` times
  the training covariance of `z` with the returns; the diagnosis's kernel
  code reproduces the panel's instrument rows; BKS at `K = L = 20` and
  `lambda = 0` without the constant's term equals the instruments only.

Columns: median OOS R2 in the forecast window and over the sweep windows
(%), and the Spearman correlation with `B_true`.

| Row | Variant | A window | A sweep | A Spearman | B window | B sweep | B Spearman |
|---|---|---|---|---|---|---|---|
| 1 | Oracle | 20.5 | 13.1 | 1.00 | 15.4 | 16.6 | 1.00 |
| 2 | OLS | -2.4 | -14.1 | 0.57 | 14.0 | 14.7 | 0.89 |
| 3 | Ridge (GCV) | 16.2 | 9.3 | 0.66 | 13.7 | 15.3 | 0.90 |
| 4 | Elastic net | 14.8 | 8.6 | 0.56 | 13.8 | 14.3 | 0.85 |
| 5 | Instruments only, training window | -2.0 | -17.1 | 0.57 | 13.9 | 14.5 | 0.89 |
| 6 | Instruments only, full history | 11.0 | 4.9 | 0.69 | 14.7 | 14.3 | 0.89 |
| 7 | Row 6 with `Sigma_z` over the same kernel-weighted history | 23.5 | 15.9 | 0.91 | 14.2 | 14.7 | 0.89 |
| 8 | Best rank 3 of row 6 | 10.7 | 5.4 | 0.66 | 10.6 | 13.4 | 0.83 |
| 9 | BKS `K = 20`, `lambda = 0` | -1.8 | -3.3 | 0.58 | 3.7 | -0.8 | 0.70 |
| 10 | BKS `K = 3`, `lambda = 0` | -0.9 | -1.7 | 0.07 | 0.8 | -0.3 | 0.26 |
| 11 | BKS `K = 3`, tuned (Compare row, full history) | -0.2 | -12.3 | 0.18 | 0.8 | -0.6 | 0.23 |
| 12 | BKS `K = 3`, tuned (Compare row, training window) | -35.8 | -42.7 | 0.08 | -1.3 | 0.4 | 0.30 |
| 13 | `B_true` through row 11's directions | -3.4 | -23.1 | 0.13 | 0.8 | 0.6 | 0.24 |

**Reasons, largest first.** Each item gives the change in window R2 and
sweep R2 (percentage points) and in Spearman, setting A then setting B.

1. **Which directions BKS keeps (the Eq. 5 step).** Row 8 to row 10
   (without the constant's term) costs -12.1 / -7.4 / -0.59 on A and
   -10.3 / -13.6 / -0.57 on B.
   - IPCA's three directions hold 3% (A) and 19% (B) of the instruments'
     squared norm at `lambda = 0`, and 38% and 37% after tuning. The best
     three hold 92% and 91%; three random directions would hold 15%.
   - IPCA's map from instruments to factor loadings works like a
     regression coefficient, so at `lambda = 0` it leans towards the
     directions where the assets differ least: on A, 2.6 of its 3
     directions lie in the instruments' 10 weakest.
   - Even `B_true`, sent through the tuned fit's directions, scores -3.4%
     (A) and 0.8% (B) (row 13).
2. **The constant term, at high `K` only.** Row 6 to row 9 costs
   -12.7 / -8.2 / -0.11 on A and -11.0 / -15.2 / -0.19 on B, up to -30
   points in single seeds; without the constant's term row 9 equals row 6.
   This is why `K = L` does not give back the instruments. At the tuned
   `K = 3` the penalty sets the constant to about zero and it costs nothing.
3. **Rank alone: small.** Row 6 to row 8 costs -0.2 / +0.5 / -0.03 on A and
   -4.1 / -0.9 / -0.06 on B. `B_true` has effective rank 6.3, and its top
   three directions hold 76% of it.
4. **Penalty and tuning: mixed.** Row 10 to row 11: +0.7 / -10.6 / +0.11 on
   A and +0.1 / -0.4 / -0.02 on B.
   - No point on the lambda path reaches a Spearman above 0.22 (A) or 0.30
     (B).
   - The in-sample Sharpe criterion (D27) is flat against its noise: on A it
     spans 2.35–2.97 against a standard error of about 1.41 over 26 weeks,
     on B 0.32–0.87 against 0.38.
   - The tuned fit on A drops 9–11 of the 20 topics, among them S1 Energy
     and A4 Financial Conditions (14% of `B_true`) in every seed. Topic
     selection is secondary: `lambda = 0` with all 20 topics is no better.
5. **Units: no loss.** The inverse-volatility conversion gains 3.4 (A) and
   0.8 (B) points against raw returns and leaves Spearman unchanged.
   Dropping the asset weighting from the tuned fit costs 12.9 (A) and 6.4
   (B) points.
6. **History: a gain, not a loss.** Row 2 to row 6 gains +13.4 / +19.0 /
   +0.12 on A, where 92% of the kernel weight lies before the training
   start; row 7 gains 25.9 points. On B the kernel lies inside the training
   window and history makes no difference.
7. **Weekly frequency: minor.** A three-direction reduced-rank regression of
   weekly returns on the instruments reproduces row 10. On daily returns it
   captures more of the instruments (10% on A, 24% on B), but its OOS R2
   stays near zero (-1.4%, 0.8%).

In total, from row 6 to row 11 BKS loses -11.1 / -17.2 / -0.51 on A and
-13.9 / -15.0 / -0.65 on B.

**The training-window variant** (row 12, D88) goes through the same step,
starting from instruments about as good as OLS (row 5). On A it scores below
the full history in each of the noise seeds 0–4; on B the two are close.

**Reading the figures.**

- A four-week window R2 moves by about ±8 points between seeds (OLS on A:
  -9.8% to 8.2%). The sweep medians are steadier.
- Seed 0, the dashboard's, is the best of three for BKS on A (7.0%, against
  1.2% and -8.7%).

**When BKS should do better.**

1. Pricing questions: which topics drive the few priced factors, the factor
   returns, a portfolio's Sharpe ratio. BKS identifies the factor betas,
   recovered at 0.97 in the simulation study (D52).
2. Topics that act on prices through a few common factors, so the Eq. 5
   step loses nothing.
3. Large cross-sections and long samples (hundreds of assets), where
   `Gamma` is precise and the topic selection stable; setting A has 55
   assets and 26 weeks.
4. Sensitivities that drift over time: BKS's instruments update weekly, while
   the lab's sensitivities are constant by design.

For asset-by-asset sensitivities, BKS's own ingredients do better in a
consistent full-history regression (row 7: 23.5% on A against 16.2% for
ridge) than through the Eq. 5 step. The lab does not offer it as a method
(TBD).

The diagnosis scripts of 2026-09-30 are outside the repository:
`diag_bks.py` runs every variant for one setting and seed, `aggregate.py`
builds the table, the step attribution, the `K` sweep and the lambda path,
and `subspace_check.py` measures where the fit's directions lie. Row 12 was
measured with `LabSession.comparison` on both BKS fits.

### G.16 BKS trace page (D90)

**Purpose.** Owner request 2026-09-30: the BKS implementation "does not
behave as expected". The page follows one lab BKS run from its inputs to its
outputs, stage by stage, and puts an independent reference next to every
result, so that a departure can be located: either the code does not do what
its formula says (an identity check is off), or the formula does something
other than what one expects (a diagnostic or a ladder row shows how far, and
against which reference). Nothing is edited on the page; the settings are the
sidebar's.

**Which run.** The BKS run of the current settings and the sidebar's
covariance history (D88), the run the BKS tab shows. The trace needs a fit
this browser session requested (D80); the page has its own Run BKS button,
the sidebar's works too, and a run on either page updates both. The
Compare methods tab links to the trace of the variant it inspects.

**Steps.** A radio chooses the step; only that step is computed and drawn.
An asset (shared with the BKS tab's per-topic split), a topic (default: the
asset's topic with the largest true sensitivity) and, where a step needs one,
a week are chosen above the steps.

| Step | What it shows | Reference ("what it should be") |
|---|---|---|
| Summary | a row per step (checks ok, key number, reading), the reference ladder, the findings, all checks | the ladder's upper rows |
| 1 Inputs | the settings in effect, the shapes, the topic's attention and the asset's return, the asset's set and true sensitivities | the population truth recomputed from the simulation's moments |
| 2 Alignment and scaling | the asset's divisor (trailing volatility, or training standard deviation) and scaled return; per-asset unit conversion | scaled return x divisor = raw return; the attention lag; the exact kernel-weighted conversion |
| 3 Attention shocks | the BKS shock against the direct estimator's shock, and its split into signal, news noise and slow component | the explicit formula; the direct shock shifted by the lead; the population attenuation and standard deviation |
| 4 Instruments | the kernel covariance of the asset and topic over every week, the kernel weights behind one week, the week's instrument row, all instruments against the population covariance | a brute-force kernel covariance; the population covariance in panel units |
| 5 Weekly panel | the pairing of each return week with the previous week's instruments, the design matrix of a week, how much the instruments move over the training weeks | the weekly sum of scaled returns; the training `sigma_c`; the training weeks from `training_weeks` |
| 6 Fit and lambda | the lambda path (criterion, standard error, tolerance floor, the null band of a Sharpe ratio with no priced signal), `Gamma` row norms along the path, the standardised `Gamma`, the KKT ratio per instrument, the in-sample factors | the objective, the factor step, KKT and scale balance, the canonical form, `lambda_max` and the band rule recomputed |
| 7 Forecast weeks | each forecast week's pooled R2 against the shuffled-instrument reference, fitted against realised returns of a week, the weekly factors | the closed-form factor and its first-order condition; the pooled R2 in panel and in exact return units |
| 8 Implied sensitivities | the Eq. 5 chain of the asset per topic (instrument, projection, constant, divisor, `Sigma_z`, standardisation), the captured share per instrument direction, implied against true sensitivities, `Sigma_z` over three samples | the step-by-step recomputation (identity) and the ladder variants |

**The reference ladder.** Each row is a topics x assets sensitivity matrix in
the direct estimator's standardised units, scored as the Compare methods tab
scores a method: Spearman correlation with `B_true` over all pairs, RMSE, and
the median over assets of the OOS R2 in the forecast window with the
sensitivities frozen at the training end. From top to bottom the rows go from
what is achievable to what BKS delivers, and the change from one row to the
next is the loss of that step:

1. True sensitivities (`B_true`).
2. The best a training window allows (the population value on the training
   window's own signal).
3. The instruments alone with `Sigma_z` over the instruments' own
   kernel-weighted history (a consistent full-history regression, G.15.1
   row 7).
4. The instruments alone with `Sigma_z` over the training days (the
   production `Sigma_z`, G.15.1 row 6).
5. The best `K` directions of the instruments.
6. The fit's `K` betas inverted by a cross-sectional least-squares
   reconstruction instead of Eq. 5 (it equals Eq. 5 when the instruments lie
   exactly in the span of `Gamma_tilde`).
7. The fit's directions without the constant's term.
8. BKS-implied, the production figure (the step-by-step recomputation equals
   it to 1e-12).

An all-zero row and the sidebar's direct method are benchmarks outside the
chain. Measured 2026-09-30 on the dashboard defaults, full history, noise
seed 0 (Spearman / median OOS R2): true 1.00 / 23.2%; window 0.78 / 25.8%;
instruments with their own `Sigma_z` 0.91 / 25.1%; with the training
`Sigma_z` 0.71 / 11.2%; best three directions 0.69 / 19.8%; least-squares
inversion of the betas 0.69; the fit's directions and production 0.19 /
7.0%. The all-zero matrix has a smaller RMSE (0.078) than production (0.091).

**Checks.** Identity checks must hold whatever the settings; an identity
check that is off points to a defect in the code. Diagnostic checks compare a
result with a reference that holds only under assumptions (the population
covariance, a live factor, instruments that track their population value);
when one is off, the run departs from the reference, which is a finding, not
a defect. Info checks report a number without a pass mark. Tolerances are
relative unless stated and set well above the numerical error of the stage
(for example 1e-12 for the unit identity, 5e-3 for the KKT ratio, whose
residual reflects the stopping rule of the alternating fit).

**Findings.** Computed from the run, in plain words, departures first:
the share of the instruments the fit's directions keep against the best `K`;
the moment windows of the full history (kernel weight before the training
start against the training days of `Sigma_z`, with the instruments' Spearman
under both); a lambda criterion within noise (its range against its standard
error, and against the 5%–95% band of the in-sample Sharpe ratio of `K`
factors with no priced signal over the training weeks); an absolute
tolerance band below a Sharpe ratio of 1; a dead factor; a path point worse
than the all-zero solution (D22); instruments that do not track their
population value; an approximate unit conversion; the week of the
instruments Eq. 5 uses. On the dashboard defaults the first trace found no
identity check off. The departures are the Eq. 5 projection, the moment
windows and the lambda choice (D90).

**Validity limits of the references.**

1. The population values pool every day of the simulation (the truth's
   full-sample moments, G.12 point 3). A six-month window departs from them
   by sampling error; the window row of the ladder is the realistic target
   for a six-month fit.
2. The population instrument in panel units divides the population
   covariance by the asset's full-sample volatility under the full history
   (the trailing volatility changes over time, so this is approximate; the
   measured correlation is 0.98 and the slope 1.07 on the defaults) and by the
   training standard deviation under the training history (exact).
3. The signal, news and slow split of the shocks is exact only without the
   attention floor (the simulation reports the clipped count).
4. The null band of the lambda criterion assumes independent weekly factor
   returns with zero means (Hotelling's `T^2`); the standard error assumes
   independent returns.
5. The per-lambda refit (the recovery panel of the lambda path) runs up to 60
   topics; above that the page says it was skipped.

**Code.** `narrative_ipca/exposure_lab/trace.py` (G.13), the session stage
`bks_trace` (G.10), the figure builders of G.13 and `dashboard/trace_page.py`;
the captions are `HOW_TRACE_*` in `dashboard/_ui.py`. Tests:
`tests/test_lab_trace.py` (every identity check holds on generic and default
configurations, both histories, both leads, fixed lambda 0 and tuned; the
chain equals production; the helpers agree with the panel) and the trace
tests in `tests/test_lab_dashboard.py`.
