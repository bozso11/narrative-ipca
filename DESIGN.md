# narrative-ipca — design and decision register

Production implementation of Bybee, Kelly & Su (2023), *Narrative Asset Pricing:
Interpretable Systematic Risk Factors from News Text* (BKS), with the estimator
theory of Kelly, Korsaye, Pruitt & Su (2026), *Instrumented Principal Component
Analysis* (KKPS). Status: v0.1, 2026-09-06 (updated after the verification round of the same day; see D46-D49). Written for a multi-asset universe
whose topic attention time series come from FASTopic and are supplied as input.

The document has six parts:

- Part A — the methodology, step by step, mapped to modules and equations.
- Part B — the decision register (D1–D51): every assumption or best guess, with
  the reason and where to change it.
- Part C — module contracts (function signatures the code implements).
- Part D — the simulation data-generating process with known ground truth.
- Part E — the evaluation harness and what the simulated data should show.
- Part F — open points and known limitations.

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
  solution); the tuner never picks them because their Sharpe is low.
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
| `placebo_selected` | number of placebo topics selected at tuned lambda | 0 (App. C.2) | ≤ 0 |
| `gamma_subspace_cos` | mean cosine of principal angles between col(Gamma_tilde_hat) and col(Gamma_tilde_true) restricted to the rows that are relevant *and selected* (non-selected rows are zero by construction and would only measure recall) | > 0.9 | ≥ 0.85 |
| `factor_canonical_corr` | first canonical correlation between `F_hat` (in-sample) and `f_period_true` | > 0.95 | ≥ 0.90 |
| `factor_canonical_corr_mean` | mean over K canonical correlations | reported | — |
| `state_canonical_corr` | first canonical correlation between `x_hat_tau` and `x_true_tau` | > 0.85 | ≥ 0.80 |
| `impact_spearman` | Spearman correlation of `I_{z->MVE}` hat vs true over the relevant topics that were selected (BKS report impact vectors for selected narratives only) | > 0.8 | ≥ 0.70 |
| `mve_sharpe_is` | in-sample MVE Sharpe of the fit | ≈ true (1.0), inflated a little | reported |
| `oos_sharpe` | realised OOS MVE Sharpe | ≈ 0.5–0.9 of true | ≥ 0.5 × true |
| `oos_sharpe_true_mve` | realised OOS Sharpe of the *true* MVE portfolio of true factors | ≈ true | reported (upper bound) |
| `systematic_r2_recovered` | R2 of true systematic return `beta f` on fitted `c Gamma_hat f_hat` (in-sample) | > 0.7 | ≥ 0.50 |
| `total_r2` | model fit | ≈ population systematic R2 | reported |
| `n_selected` | | ≈ n_relevant | reported |
| `instrument_beta_r2_*` | cross-sectional R2 of the instrument `cov[t, :, l]` on the true `beta_t` (mean over sampled periods), split into relevant / noise / placebo topics; `_chance` = K / N | relevant ≫ noise > chance (baseline); noise > chance (topic_null); all ≈ chance (no_factor) | reported |
| `oos_selection_stability` | mean Jaccard similarity of the selected set between consecutive refits | high in baseline *and* topic_null (the spurious instruments persist with the kernel), low only in no_factor; a diagnostic of estimation noise, not of narrative information (D47) | reported |
| `null_selection_lift` (no_factor only) | n_selected / round(0.05 L), the number selected relative to a 5% chance level | ≈ 1, no lift | ≤ 2.0 |
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
