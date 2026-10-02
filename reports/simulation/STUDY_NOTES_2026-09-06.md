# Simulation study — what the estimator recovers, and what it does not (2026-09-06)

Full-size simulation of the BKS pipeline on data with known characteristics:
500 assets in four asset classes, 120 topics (20 carry signal, 20 are
variance-matched placebos, 80 are persistent noise), K = 3 states, 20 years of
daily data, monthly periods, 40% of the sample out of sample with annual
refits. Five scenarios, three seeds each, three lambda-tuning rules:
`bks` (in-sample MVE Sharpe, exact argmax, the paper's rule), `tol02` (same
criterion, sparsest point within 2% of the maximum), `loocv` (leave-one-period-out
Sharpe, BKS App. C.3, 16 subsampled folds). Tables: `full/STUDY_2026-10-02.md`;
full detail per variant: `full/<variant>/harness_<date>.md` (`bks` and `loocv`
2026-09-06, `tol02` 2026-10-02). Metric definitions and thresholds:
`DESIGN.md` Part E; decisions D47 and D52 explain the two findings below.

**Update 2026-10-02: `tol02` re-run after the D51 fix.** The 2% band is now
relative at every level of the Sharpe ratio; before, it was 0.02 in absolute
terms whenever the best in-sample Sharpe was below 1. A re-run with the old
rule gives the 2026-09-06 selections, pass flags and reported figures (stored
values agree to 1e-13), so every change below is the fix. Both runs used
`--out` in a scratch directory, side by side on 4 workers each; the new run's
outputs were copied to `full/tol02/`, with the paths in its report and in
`full/study_run.log` rewritten to that directory. Its runtimes are therefore
not comparable with the `bks` and `loocv` columns of 2026-09-06.

1. 5 of 15 runs change. `no_factor` seed 2 and `weak` seed 1 choose a
   smaller lambda (3 → 5 and 112 → 116 topics); `no_factor` seed 1,
   `topic_null` seed 1 and `weak` seed 0 change only through the retuning at
   the out-of-sample refits.
2. `baseline` and `softmax` are unchanged, so finding 3 and the
   recommendation below stand.
3. On `no_factor` the full-sample band now admits only the argmax, so
   `tol02` selects what `bks` selects there (4.7 topics, 1.3 placebos); the
   out-of-sample refits still differ slightly (OOS Sharpe -0.23 against
   -0.24).
4. `weak`: 58 topics and 8.7 placebos on average (was 57 and 8.0); the OOS
   ratio check passes in 1 of 3 seeds (was 2), so no `weak` seed passes every
   check. The table below has the new figures.

## Key findings

1. **The factor structure is recovered in every signal scenario, under every
   tuning rule.** Implied betas `c Gamma` reach a first canonical correlation of
   0.97 with the true betas, the factors 0.99, the fitted systematic return
   explains 0.93 of the true one, and the out-of-sample MVE portfolio earns
   about 70% of the Sharpe ratio the true factors' MVE portfolio realises over
   the same months (0.69 vs 0.95, three seeds). The in-sample MVE Sharpe
   (1.11) sits close to the population value (1.0). The simplex mapping of
   attention (`softmax`) changes none of this.

2. **The estimator finds priced factor portfolios even when no topic carries
   information (`topic_null`).** With `A = 0` and priced factors in returns,
   the kernel covariances of noise topics still span the assets' betas
   (cross-sectional R2 on the true betas 0.105 against a chance level of
   0.007), the fit recovers betas at 0.92 and factors at 0.98, and the
   out-of-sample MVE earns 0.39 (40% of the true MVE's 0.95), with a selected
   set as stable across refits as in baseline (0.83 vs 0.85). An OOS Sharpe
   ratio therefore cannot certify that narratives carry information, and
   neither can selection above chance or stability (D47). The chance-level
   null (`no_factor`) behaves as expected: 3-6 topics selected against a
   chance level of 6, OOS Sharpe within one standard error of zero, and an
   in-sample Sharpe of 0.5, which is the optimism of three noise factors over
   130 training months.

3. **The paper's tuning rule is not selective on this panel.** The in-sample
   Sharpe surface is flat from about 90 down to about 10 selected topics
   (1.05 to 1.06), so the exact argmax lands on a dense point: 91 topics
   selected on average, precision 0.19, 15 of 20 placebos in. The 2%
   tolerance rule selects 9-11 topics with precision 1.0 and no placebo in two
   of three seeds (the third seed lands in the dense region), with the same
   OOS Sharpe. LOOCV is erratic (3 to 118 topics) and six times slower. BKS's
   own placebo result (no placebo selected at the tuned lambda) is reproduced
   here only under a sparsity-preferring rule, not under the argmax.

4. **Row-level loadings, the impact vector and the latent states are not
   identified by the model** (D52): Eq. 5 makes the population instruments
   rank K, so any `Gamma_tilde` in an (L − K)-dimensional family reproduces
   every beta, and the group lasso returns one sparse representative. The
   harness measures this directly: principal-angle cosines of about 0.6
   between estimated and true loading rows and impact-vector rank
   correlations of 0.45-0.65, next to 0.97-0.99 for betas and factors. These
   three metrics are reported, not pass/failed. For the research plan this
   means the BKS interpretation map (which topics, by how much) is a property
   of the representative the lasso picks; per-topic attribution needs the
   separately identified topic-sensitivity layer of the plan.

5. **Runtime.** One full run (tune on 20 lambdas, 8 annual OOS refits with
   retuning, wrap-up, evaluation) takes 5-7 minutes on three BLAS threads with
   the numba kernel (D46); the pure-Python group lasso took 19 minutes for the
   in-sample tune alone. LOOCV runs take 30-50 minutes.

## Expected vs observed, per scenario (means over three seeds)

| scenario | rule | selected | recall (strong) | precision | placebos | beta CC1 | factor CC1 | sys. R2 | OOS Sharpe / true | expected | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | bks | 91 | 0.83 | 0.19 | 14.7 | 0.974 | 0.993 | 0.94 | 0.69 / 0.95 | factors and betas recovered, few placebos | factors yes; selection dense (finding 3) |
| baseline | tol02 | 34 | 0.70 | 0.73 | 4.7 | 0.975 | 0.994 | 0.94 | 0.69 / 0.95 | as above | yes in 2 of 3 seeds |
| baseline | loocv | 43 | 0.57 | 0.72 | 6.3 | 0.974 | 0.994 | 0.91 | 0.62 / 0.95 | as above | erratic selection |
| softmax | bks | 50 | 0.80 | 0.49 | 6.7 | 0.975 | 0.994 | 0.94 | 0.63 / 0.95 | same as baseline | same as baseline |
| weak | bks | 84 | 0.93 | 0.44 | 13.3 | 0.969 | 0.992 | 0.91 | 0.28 / 0.55 | graceful degradation | factors yes; two seeds select all 120 (argmax at the dense end) |
| weak | tol02 | 58 | 0.93 | 0.51 | 8.7 | 0.971 | 0.993 | 0.92 | 0.25 / 0.55 | graceful degradation | factors yes; seed 1 dense (116 topics, 20 placebos); OOS ratio passes in 1 of 3 seeds |
| no_factor | bks | 4.7 | n/a | n/a | 1.3 | n/a | 0.20 | n/a | −0.24 (se 0.37) | chance level | yes |
| no_factor | loocv | 74 | n/a | n/a | 13 | n/a | 0.22 | n/a | −0.28 (se 0.37) | chance level | selection arbitrary on pure noise |
| topic_null | bks | 84 | n/a | n/a | 14.7 | 0.921 | 0.976 | 0.86 | 0.39 / 0.95 | positive, degraded (D47) | as predicted |

"strong" recall is the recall over the ten relevant topics with the larger
`||A_l||`; "sys. R2" is the share of the true systematic return explained by the
fitted values; "OOS Sharpe / true" is the realised annualised Sharpe of the
estimated MVE over the 88 OOS months next to that of the true factors' MVE over
the same months.

## What this says about the production design

- The estimation chain (shocks, kernel covariances, panel, Sparse IPCA, OOS)
  behaves as the paper describes and recovers the tradable objects. It is
  ready to be run on the FASTopic attention series and the multi-asset
  returns; every input assumption is listed in `DESIGN.md` D1-D8.
- The lambda rule needs a decision before production use. I suggest the
  tolerance rule (`TuningConfig.tolerance = 0.02`) as the default for the
  pilot, with the argmax kept as the BKS reference run, but open to challenge:
  the tolerance value is a judgement call and should be re-examined on the
  pilot data (TBD in the pilot).
- Evidence that narratives carry information has to be relative: the placebo
  test (real topics must beat variance-matched placebos at the tuned lambda),
  pricing errors on test assets, and a topic-null comparison. Sharpe ratios,
  selection counts and selection stability do not discriminate (finding 2).
- Topic-level interpretation from this model (impact vectors, states) should
  be read as a description of the selected representative, not as an estimate
  of the data-generating loadings (finding 4).

## Limitations of the study

1. Three seeds per scenario; the seed-to-seed spread of the OOS Sharpe (0.36
   to 0.92 in baseline) is dominated by the 88-month evaluation window, as it
   would be in real data.
2. The DGP is the BKS model itself (Figure 1 of the paper) with Gaussian
   shocks and a linear attention mapping; it has no attention-return
   simultaneity, no topic drift and no structural breaks (Part F).
3. The dense end of the lambda grid (1e-4 of `lam_max`) hits the 500-sweep
   cap; those points are reported with `converged = False` and still won the
   argmax in three weak/baseline seeds.
4. The pass/fail thresholds are best guesses (D40); the report shows every
   miss with the observed value, and the two checks that turned out to be
   ill-posed (loading rows, impact vector, states; placebos under `no_factor`)
   were demoted with the reason recorded (D52).

## Regenerate

```bash
python scripts/run_full_study.py --variants bks,tol02,loocv --seeds 3 --workers 8 --out reports/simulation/full
```

```bash
python scripts/summarise_study.py --out reports/simulation/full
```
