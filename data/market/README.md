# Market data: 55 lab assets (2015–2025)

Daily returns for the 55 assets of the topic-exposure lab (DESIGN.md Part G), built from 58 **legs** (one proxy series each). Every asset "A v B" is long leg A and short leg B. The outright assets (Global Duration, Global Equity, Global Credit, USD) and every "XXX v USD" pair are long against the **cash** leg, which has a return of 0.

Status of the run on 2026-09-29: all 58 legs are `ok` (no missing level in the delivered window). Three FX legs come from FRED (CNY as the fallback, TWD pinned, and the JPY conversion of the JP 7-10y leg). No leg failed, so the lab uses no artificial fill for the listed assets.

**Regenerate** (from the repository root, needs the `lab` extra; about 20 seconds online):

```bash
.venv/Scripts/python.exe scripts/fetch_market_data.py
```

Registries are read from `data/reference/` (`legs.csv`, `assets.csv`) and outputs go to `data/market/`; `--offline` rebuilds from `raw/` without the network. The CSV copies of the return tables are written too but not committed.

## Key open items

1. Confirm the (TBC) readings in Section 4, in particular Global Equity = ACWI, USD = DXY and IT 7-10y = Italy.
2. Decided 2026-09-29 (TBC): the JP 7-10y leg is converted to JPY with FRED noon fixings (`--conversion-fx fred`, now the default), because they are closer in time to the Xetra close than Yahoo's midnight-London snapshot. JP 7-10y v Global Duration vol falls from 7.7% to 6.0%.
3. Non-synchronous daily closes inflate daily spread volatility (Section 6, item 3). The lab keeps daily returns for its forecast windows (a 1–12 week window would otherwise hold 1–12 observations) and runs BKS on weekly periods; DESIGN.md G.12 lists this as a limitation.
4. Decided 2026-09-29 (TBC): TWD is pinned to FRED DEXTAUS in `legs.csv`, with Yahoo as the fallback, because Yahoo TWD sits at the quality gate and a rerun could switch back to its poor 2015–16 data.
5. Open: whether Friday FX levels should come from another snapshot. After re-dating, Friday is Yahoo's Sunday-evening quote, so weekend gaps land on Friday (Section 8, remaining item 2).

## 1. Files

| file | content |
|---|---|
| `scripts/fetch_market_data.py` (repository) | Standalone, re-runnable builder. Reads the two registries in `data/reference/`, downloads, cleans, aligns and writes everything below. |
| `data/reference/legs.csv` | Leg registry: `leg_id`, name, benchmark index, proxy ticker, source, fallbacks, quote and return currency, TR or price, invert flag, notes, plus `leg_type` and `components` (for the constructed basket). |
| `data/reference/assets.csv` | Asset registry, 55 rows in the order of the source image: `order`, `asset_id`, `name`, `long_leg`, `short_leg`, `asset_class`, `sub_class`. |
| `raw/<ticker-or-series>.csv` | Each series as downloaded (`date`, `close`, `adj_close` for Yahoo). No cleaning is applied here. |
| `leg_levels.parquet` | Weekday calendar x `leg_id`: level in the leg's return convention, 2014-12-31 to 2025-12-31 (2,871 rows). |
| `leg_stale.parquet` | Boolean mask, same shape: `True` where the level was forward-filled (no fresh observation that day). |
| `leg_returns.parquet`, `leg_returns.csv` | Simple daily leg returns, 2015-01-02 to 2025-12-31 (2,869 rows x 58). |
| `asset_returns.parquet`, `asset_returns.csv` | 55 columns named by `asset_id`, long-leg return minus short-leg return, same dates. |
| `manifest.json` | Per leg: source and ticker actually used, first/last valid date, `n_valid`, `n_stale`, `n_missing`, `fallback_used`, `status`, candidates tried and diagnostics. Also per-asset coverage and annualised vol, the download log, run timestamp and package versions. |
| `.cache/yfinance/` | yfinance timezone cache (safe to delete). |

## 2. Conventions

1. **Calendar**: all weekdays (Mon–Fri), no holiday calendar. Levels start 2014-12-31; returns run 2015-01-02 to 2025-12-31 inclusive. The 2015-01-01 return row is outside the requested range, so compounding the returns from the 2014-12-31 level misses any 1-Jan move (zero for all legs except a few FX legs).
2. **Forward-fill**: a leg level is forward-filled over weekdays with no fresh observation for at most 5 consecutive weekdays (pandas limit semantics: in a longer gap the first 5 are filled and the rest stay NaN). `leg_stale` marks every filled cell, and the return on a stale day is 0.
3. **Returns**: r_t = L_t / L_{t−1} − 1, where L_t is the leg level on weekday t. Asset return = r_long − r_short.
4. **Price field**: ETFs use Yahoo adjusted close (total return, dividends reinvested). The script downloads with `auto_adjust=False` so that `raw/` keeps both `close` and `adj_close`; `adj_close` is the same number `auto_adjust=True` returns as Close. Indices and FX use close.
5. **Equity legs**: US-listed ETFs in USD, except Quality (IWQU.L) and Value (IWVL.L), which are the USD lines of the LSE-listed UCITS funds.
6. **Rates legs**: the country leg is in local currency (IEF USD, IGLT.L GBP, EXHD.DE EUR, IITB.MI EUR, CSBGC0.SW CHF, XJSE.DE converted to JPY). Global Duration is DBZB.DE (EUR-hedged).
7. **Credit legs**: CORP.L (USD), IEAC.L (EUR, local), EMLC (USD).
8. **FX legs**: spot only, no carry. The level is the USD price of one unit of XXX, so the return of "XXX v USD" is the % change of that price. Yahoo `XXX=X` quotes (XXX per USD) are inverted. FRED H.10 (`DEXxxUS`, noon New York) is the fallback.
9. **Currency conversion** (non-FX legs whose quote currency differs from the return currency): level_ret = level_quote × usd_per(quote) / usd_per(return), where usd_per(XXX) is the level of leg `fx_xxx`. Conversion happens on the proxy's own trading days, before the forward-fill, so an exchange holiday does not create an FX-only return. The only leg converted in this run is `gov_jp_7_10` (EUR → JPY).
10. **EM FX basket** (`fx_em_basket`): equal-weight, daily-rebalanced mean of the 12 component spot returns (ZAR, MXN, BRL, COP, CLP, KRW, CNY, IDR, THB, CZK, TWD, INR), with at least 8 of 12 required on a day; level rebased to 100 on 2014-12-31. A basket day is stale only when no component is fresh. CEW is downloaded as `fx_em_cew_ref` for reference and is not used in any asset.
11. **Fallbacks**: the proxy is tried first, then `fallback_tickers` in order. The first candidate with no missing level in the window (and, for FX, passing the quality gate in item 14) is used for the whole series; there is no splicing. Token grammar: `[yahoo:|fred:]TICKER[@QUOTE_CCY][/inv]`, for example `fred:DEXSZUS/inv` or `yahoo:ISF.L@GBp`.
12. **FX dates**: Yahoo `=X` daily bars are re-labelled one weekday earlier. Yahoo stores the snapshot taken at the start of the labelled day (about 00:00 London, the end of the previous New York day). The evidence is below; `--no-fx-redate` switches this off.
    - Same-day correlation of Yahoo daily returns with the US-listed currency ETFs FXE, FXY and FXF is 0.04–0.07; after the one-day shift it is 0.92–0.96.
    - The SNB floor removal (Thursday 2015-01-15) and the BoJ yield-curve change (2022-12-20) appear on the next day's Yahoo bar.
    - After the shift, the correlation of Yahoo with FRED daily returns peaks at lag 0 for every pair (manifest `corr_daily_ret_by_lag`).
    - DX-Y.NYB is correctly dated (same-day correlation with UUP is 0.95) and is not shifted.
13. **FX cleaning** (Yahoo FX legs only; `--no-fx-clean` switches it off):
    - A print identical to the previous one is a vendor carry-forward. It is dropped and becomes an ordinary stale day.
    - An isolated bad tick is dropped. That is a print that sits far from both neighbours and reverses the next day; the threshold is max(6 robust sigmas, 2%) (8 before the QA in Section 8; `--fx-bad-tick-sigma 8` restores it). Examples (raw Yahoo dates, ZAR and NOK per USD): ZAR 14.86 on 2024-11-15 between 18.23 and 18.17; NOK 7.74 on 2020-03-20 between 10.78 and 11.76.
    - 100x decimal slips are rescaled (COP 6, CLP 1). The same 100x rule runs on all Yahoo series to catch GBp/GBP mix-ups on LSE listings; it found none on the ETFs.
    - Every dropped or rescaled date is listed in the manifest (`bad_ticks_removed`, `repeated_prints_dropped`, `fixes_100x`).
14. **FX quality gate**: when a FRED fallback exists, Yahoo is compared with it after cleaning. If the weekly-return correlation is below 0.8, Yahoo counts as failed and FRED is used (`quality_gate` in the manifest).
15. **Failures**: a failed ticker is recorded in `manifest.json` (`downloads`, `summary.failed_downloads`) and does not stop the run. A leg with no usable candidate is left NaN and marked `failed`; nothing is synthesised here. Artificial fill belongs to the dashboard code.

## 3. Legs and coverage

Coverage is over the 2,871 level dates. "Stale" counts forward-filled weekdays, mostly exchange holidays.

| leg | benchmark index | proxy (source used) | return ccy | coverage |
|---|---|---|---|---|
| eq_world | MSCI World (NR) | URTH (Yahoo) | USD | ok, 104 stale |
| eq_acwi | MSCI ACWI (NR) | ACWI (Yahoo) | USD | ok, 104 stale |
| eq_em | MSCI Emerging Markets | EEM (Yahoo) | USD | ok, 104 stale |
| eq_energy | MSCI World Energy | IXC (Yahoo) | USD | ok, 104 stale |
| eq_materials | MSCI World Materials | MXI (Yahoo) | USD | ok, 104 stale |
| eq_industrials | MSCI World Industrials | EXI (Yahoo) | USD | ok, 104 stale |
| eq_cons_disc | MSCI World Consumer Discretionary | RXI (Yahoo) | USD | ok, 104 stale |
| eq_cons_staples | MSCI World Consumer Staples | KXI (Yahoo) | USD | ok, 104 stale |
| eq_healthcare | MSCI World Health Care | IXJ (Yahoo) | USD | ok, 104 stale |
| eq_financials | MSCI World Financials | IXG (Yahoo) | USD | ok, 104 stale |
| eq_info_tech | MSCI World Information Technology | IXN (Yahoo) | USD | ok, 104 stale |
| eq_comm_services | MSCI World Communication Services | IXP (Yahoo) | USD | ok, 104 stale |
| eq_utilities | MSCI World Utilities | JXI (Yahoo) | USD | ok, 104 stale |
| eq_real_estate | MSCI World Real Estate | RWO (Yahoo) | USD | ok, 104 stale |
| eq_switzerland | SMI or MSCI Switzerland (TBC) | EWL (Yahoo) | USD | ok, 104 stale |
| eq_emu | MSCI EMU | EZU (Yahoo) | USD | ok, 104 stale |
| eq_uk | MSCI UK | EWU (Yahoo) | USD | ok, 104 stale |
| eq_japan | MSCI Japan | EWJ (Yahoo) | USD | ok, 104 stale |
| eq_australia | MSCI Australia | EWA (Yahoo) | USD | ok, 104 stale |
| eq_us_large | S&P 500 / MSCI USA (TBC) | IVV (Yahoo) | USD | ok, 104 stale |
| eq_us_smid | Russell 2500 / MSCI USA SMID (TBC) | VXF (Yahoo) | USD | ok, 104 stale |
| eq_quality | MSCI World Sector Neutral Quality | IWQU.L (Yahoo) | USD | ok, 91 stale |
| eq_value | MSCI World Enhanced Value | IWVL.L (Yahoo) | USD | ok, 91 stale |
| dur_global | FTSE World Government Bond Index, EUR-hedged [VERIFY] | DBZB.DE (Yahoo) | EUR | ok, 78 stale |
| gov_us_7_10 | ICE US Treasury 7-10 Year | IEF (Yahoo) | USD | ok, 104 stale |
| gov_uk_7_10 | FTSE Actuaries UK Gilts 7-10y (TBC) | IGLT.L (Yahoo), all maturities | GBP | ok, 92 stale |
| gov_de_7_10 | eb.rexx Government Germany 5.5-10.5 | EXHD.DE (Yahoo) | EUR | ok, 79 stale |
| gov_it_7_10 | Italy BTP 7-10y | IITB.MI (Yahoo), all maturities | EUR | ok, 82 stale |
| gov_ch_7_10 | SBI Domestic Government 7-15 | CSBGC0.SW (Yahoo) | CHF | ok, 108 stale |
| gov_jp_7_10 | JGB 7-10y | XJSE.DE (Yahoo), all maturities, EUR → JPY with FRED noon fixings | JPY | ok, 77 stale |
| cr_global | Bloomberg Global Aggregate Corporate (USD, unhedged) | CORP.L (Yahoo) | USD | ok, 91 stale |
| cr_eur_ig | Bloomberg Euro Aggregate Corporate | IEAC.L (Yahoo) | EUR | ok, 91 stale |
| emd_local | J.P. Morgan GBI-EM Global Diversified | EMLC (Yahoo) | USD | ok, 104 stale |
| usd_dxy | ICE US Dollar Index (DXY) | DX-Y.NYB (Yahoo) | USD | ok, 130 stale |
| fx_em_basket | MSCI EM Currency Index (proxy) | 12 EM fx legs (constructed) | USD | ok, 0 stale |
| fx_chf | CHF/USD spot | CHF=X (Yahoo) | USD | ok, 12 stale |
| fx_eur | EUR/USD spot | EURUSD=X (Yahoo) | USD | ok, 18 stale |
| fx_gbp | GBP/USD spot | GBPUSD=X (Yahoo) | USD | ok, 13 stale |
| fx_jpy | JPY/USD spot | JPY=X (Yahoo) | USD | ok, 11 stale |
| fx_cad | CAD/USD spot | CAD=X (Yahoo) | USD | ok, 11 stale; 1 bad tick dropped |
| fx_aud | AUD/USD spot | AUDUSD=X (Yahoo) | USD | ok, 14 stale |
| fx_nzd | NZD/USD spot | NZDUSD=X (Yahoo) | USD | ok, 12 stale |
| fx_sek | SEK/USD spot | SEK=X (Yahoo) | USD | ok, 8 stale |
| fx_nok | NOK/USD spot | NOK=X (Yahoo) | USD | ok, 14 stale; 5 bad ticks dropped |
| fx_zar | ZAR/USD spot | ZAR=X (Yahoo) | USD | ok, 11 stale; 3 bad ticks dropped |
| fx_mxn | MXN/USD spot | MXN=X (Yahoo) | USD | ok, 10 stale |
| fx_brl | BRL/USD spot | BRL=X (Yahoo) | USD | ok, 20 stale |
| fx_cop | COP/USD spot | COP=X (Yahoo; no FRED series) | USD | ok, 42 stale; 3 bad ticks dropped, 6 100x fixes |
| fx_clp | CLP/USD spot | CLP=X (Yahoo; no FRED series) | USD | ok, 51 stale; 1 bad tick dropped, 1 100x fix |
| fx_krw | KRW/USD spot | KRW=X (Yahoo) | USD | ok, 25 stale |
| fx_cny | CNY/USD spot | **DEXCHUS (FRED fallback)** | USD | ok, 122 stale. Yahoo CNY=X repeats one quote on 10 consecutive bars (2025-05-23 to 2025-06-05, raw dates), i.e. a gap > 5 weekdays |
| fx_idr | IDR/USD spot | IDR=X (Yahoo; no FRED series) | USD | ok, 49 stale; 6 bad ticks dropped |
| fx_thb | THB/USD spot | THB=X (Yahoo) | USD | ok, 68 stale; 7 bad ticks dropped |
| fx_czk | CZK/USD spot | CZK=X (Yahoo; no FRED series) | USD | ok, 8 stale |
| fx_twd | TWD/USD spot | **DEXTAUS (FRED, pinned 2026-09-29)** | USD | ok, 122 stale. Pinned because Yahoo TWD=X sits at the quality gate (weekly corr with FRED 0.7993 after cleaning; Yahoo vol about twice FRED's in 2015–2016); Yahoo is the fallback |
| fx_inr | INR/USD spot | INR=X (Yahoo) | USD | ok, 22 stale; 1 bad tick dropped |
| fx_em_cew_ref | WisdomTree Emerging Currency Strategy (reference only) | CEW (Yahoo) | USD | ok, 104 stale |
| cash | none | none | USD | ok (level 1, return 0) |

FRED legs have more stale days because H.10 skips US federal holidays.

## 4. Assumptions (TBC)

1. "IT 7-10y" is Italy government bonds, not Information Technology (TBC).
2. "EM v USD" is an EM currency basket; "EM v World EQ" is EM equity (TBC).
3. "Global Equity" is MSCI ACWI (proxy ACWI); "World EQ" is MSCI World (proxy URTH) (TBC).
4. "USD" outright is the ICE US Dollar Index (DX-Y.NYB), with FRED DTWEXBGS (broad trade-weighted dollar, a different index) as fallback (TBC).
5. The EM FX basket is the equal-weight, daily-rebalanced basket of the 12 EM currencies in the asset list, spot vs USD (TBC). Weekly correlation with CEW is 0.92 and annualised vol is 6.7% vs 7.0%.
6. FX legs are spot returns without carry; FRED H.10 is the fallback when Yahoo fails or has a gap > 5 business days (TBC).
7. Equity legs are USD-listed ETFs on adjusted close; rates country legs are in local currency; Global Duration is DBZB.DE, EUR-hedged; credit legs are CORP.L (USD), IEAC.L (EUR) and EMLC (USD); Quality is IWQU.L and Value is IWVL.L (TBC).
8. Adjusted close (`auto_adjust=True` equivalent) for ETFs; close for indices and FX (TBC).
9. Yahoo `=X` FX bars are re-dated one weekday earlier (Section 2, item 12). This was not in the brief; the evidence is strong, but please confirm (TBC).
10. Yahoo FX cleaning: repeated prints and isolated bad ticks are dropped, 100x slips rescaled (Section 2, item 13). This was not in the brief (TBC).
11. FX quality gate at weekly correlation 0.8 with FRED counts as "Yahoo fails" (Section 2, item 14). It switches only TWD, at 0.7993 (TBC). The margin is thin; see Section 8.
12. The JP leg is converted with FRED noon fixings (`--conversion-fx fred`, the default since 2026-09-29; TBC; see Section 6, item 4). `--conversion-fx legs` uses the re-dated Yahoo EURUSD=X and JPY=X instead.

## 5. Sanity checks from this run

1. All 58 legs are complete over the window; no NaN in levels or asset returns.
2. `asset_returns` equals `leg_returns[long] − leg_returns[short]` exactly, for all 55 assets.
3. Implied yearly distribution yields from `adj_close / close` look like coupons for every UCITS Dist fund. Examples: IGLT.L 0.8–2.1% in 2015–2022, rising to 4–6% in 2024–2025; CORP.L 2.1–4.1%; IEAC.L 0.8–3.5%. So Yahoo does apply their dividends, at least at yearly resolution.
4. Yahoo's currency metadata matches the registry `quote_ccy` for every Yahoo ticker used (IGLT.L is quoted in GBP, not GBp).
5. Annualised vol per asset is in `manifest.json` → `assets.<id>.ann_vol` (daily returns × √261).

## 6. Known limitations

1. **Spot FX without carry**: every FX asset and the EM basket ignore the interest differential, which matters most for BRL, MXN, ZAR, COP, IDR and INR. CEW (with carry) is included for comparison only.
2. **Maturity mismatch**: UK (IGLT.L), IT (IITB.MI) and JP (XJSE.DE) use all-maturity government bond funds, so their duration is longer than the 7-10y bucket. DE is 5.5–10.5y and CH is 7–15y.
3. **Non-synchronous closes**: legs close at different times. Equity ETFs are at the US close (16:00 New York). UCITS listings are at the London, Xetra, Milan and SIX closes. Yahoo FX, after re-dating, is at about the end of the New York day, and FRED is at noon New York. Daily spreads between legs from different closes carry timing noise, as the table shows. Weekly returns remove most of it.

   | asset | daily ann. vol | weekly ann. vol |
   |---|---|---|
   | Quality v World EQ (IWQU.L v URTH) | 14.5% | 6.4% |
   | Value v World EQ (IWVL.L v URTH) | 15.2% | 9.4% |
   | US 7-10y v Global Duration (IEF v DBZB.DE) | 5.1% | 3.4% |
   | JP 7-10y v Global Duration | 7.7% | 5.0% |

4. **JPY conversion of XJSE.DE**: the Xetra close (17:30 CET) and the Yahoo FX snapshot are about 6 hours apart, so FX noise enters the JPY level. The JP leg has 7.6% daily-annualised vol with Yahoo FX, 5.5% with FRED noon fixings (30 minutes from the Xetra close), and 12.5% before the Yahoo re-dating. A JGB all-maturity index should be well below any of these, so this leg is the noisiest in the set.
5. **UCITS Dist adjusted close**: yearly distribution totals look right (Section 5, item 3), but individual ex-dates and amounts are not checked against the providers' total-return series [VERIFY].
6. **Communication Services (IXP)**: the sector was Telecom Services until the GICS change in September 2018, so the series has a structural break then.
7. **Financials (IXG)** included Real Estate until the GICS change in 2016 (Real Estate became its own sector in September 2016). The Real Estate v World EQ and Financials v World EQ spreads overlap before then.
8. **Sector proxies** track S&P Global 1200 sectors, which include some EM names, not MSCI World sectors.
9. **Residual Yahoo data issues** that the cleaning does not fix:
   - CLP=X is flat at about 1050 per USD on 2022-07-18 to 2022-07-20, then drops to 927 on 2022-07-21 (raw Yahoo dates). That is a 13% one-day CLP gain. It looks like a stale feed around the July 2022 FX intervention by Chile's central bank [VERIFY]. The weekly total is probably right, but the daily path is not.
   - THB=X in 2015–2016 and CNY=X in 2017 and 2023 are noisier than FRED (weekly corr 0.87–0.88 over the window after the QA cleaning). Both passed the 0.8 gate; THB stays on Yahoo, and CNY moved to FRED for another reason (Section 3).
   - IITB.MI has seven identical adjusted closes from 2022-09-12 to 2022-09-20 (no trades or a stale feed).
10. **Year-end FX**: Yahoo has no bar labelled 2026-01-01, so the 2025-12-31 FX level is carried forward from 2025-12-30 (stale). The last FX move of 2025 is therefore not in the data.
11. **Hedging**: Global Duration is EUR-hedged while the country legs are local-currency returns. That is an acceptable approximation for duration spreads, but it is not a USD-hedged benchmark. Global Credit (CORP.L) is unhedged USD.

## 7. How to regenerate

Use the nap-lab interpreter (Python 3.13, pandas 3.0, yfinance 1.7, pyarrow):

```
C:/Users/bozso/repos/nap-lab/.venv/Scripts/python.exe fetch_market_data.py
```

A full run takes about 20 seconds and overwrites every output in this folder. Useful options:

1. `--out DIR`, `--start 2014-12-01`, `--end 2026-01-05`: output folder and download window (Yahoo `end` is exclusive).
2. `--first-level`, `--first-return`, `--last-date`, `--ffill-limit`: delivered window and forward-fill limit.
3. `--offline`: rebuild all outputs from `raw/*.csv` without network access.
4. `--conversion-fx fred`: convert the JP leg (and any `@CCY` fallback) with FRED noon fixings.
5. `--no-fx-redate`, `--no-fx-clean`, `--fx-min-weekly-corr X`, `--no-fred-check`: switch off the FX date fix, the cleaning, or the quality gate and cross-check. `--fx-bad-tick-sigma 8` reproduces the pre-QA cleaning.
6. `--legs`, `--assets`: alternative registries. To change a proxy, edit `legs.csv`; the script contains no tickers.

## 8. QA 2026-09-29

An independent QA pass recomputed every check below from the output files, `raw/` and the registries, without relying on the build log. Two code fixes followed, and all outputs were regenerated with a full online run.

### Checks run and results

| # | check | result |
|---|---|---|
| 1 | `assets.csv`: 55 rows, `order` 1–55 in image order, names identical to the asset-legs report (Section 6 there), every `long_leg`/`short_leg` in `legs.csv` and matching the report's long/short labels, `asset_class` only Equity/FX/Fixed income, `asset_returns` columns in the same order | pass |
| 2 | FX conventions: USD-per-unit levels on 2015-01-02 and 2025-12-31 plausible for all 21 pairs and DXY (e.g. CHF 0.994 and 1.263, JPY 0.0083 and 0.0064, KRW 0.00090 and 0.00070); staged level / FRED H.10 level has a median within 0.06% of 1 for all 17 pairs with FRED; every FX leg has a negative weekly correlation with DXY; JPY v USD falls in each year 2021–2024 (−34% in total); CHF v USD is +19.3% on 2015-01-15 | pass |
| 3 | Returns sanity: moves above 15%, vol by class, spreads vs legs, March 2020 | pass after fix 1 |
| 4 | Coverage 2014-12-31 to 2025-12-31: NaN and stale share per leg, gaps above 5 weekdays | pass |
| 5 | Asset return = long-leg return − short-leg return, recomputed | pass (exact) |
| 6 | JP 7-10y conversion: correlation with JPY v USD | pass |
| 7 | Rerun from scratch into `_rerun_check` | pass within float noise (see below); folder deleted |

**Check 3 detail.**
1. Leg moves above 15% in one day: CHF 2015-01-15 (+19.3%, SNB), Energy 2020-03-09 (−19.5%) and 2020-03-24 (+17.3%), Australia 2020-03-16 (−16.1%). All are genuine. No asset except CHF v USD moves more than 15%.
2. Moves of 10–15% are Brexit (EMU and UK equity on 2016-06-24), March 2020 equity, Energy on 2020-11-09 and the 2025-04-09 tariff pause, all genuine. Two are data errors: COP +10.1% then −10.1% on 2015-08-10/11 (removed by fix 1) and CLP +13.3% on 2022-07-20 (stale-then-jump, remaining item 4).
3. No 100x unit switch or split in any ETF `close`; the largest one-day jump in any ETF raw close is below 25%. IGLT.L is in GBP throughout (9.56–15.16).
4. Annualised daily vol: equity legs 13.7–26.9%, FX legs 4.0–17.5%, government bond legs 4.6–9.0%, credit legs 4.3–10.2%.
5. Every equity spread has lower vol than its long leg. Two fixed income spreads do not: EUR IG v Global Credit (5.2% vs 4.3%) and JP 7-10y v Global Duration on daily data (7.7% vs 7.6%; weekly 5.0% vs 5.2%). See remaining item 7.
6. March 2020 (2020-02-19 to trough): every equity leg falls 24–55%; Global Credit −12.9%, EUR IG −14.0%, EMD local −20.3%; IEF does not fall.

**Check 4 detail.** No leg has a NaN level or return. Stale share is 3.6% for US listings (4.5% for DXY), 2.7–3.8% for European listings, 0.3–2.4% for Yahoo FX and 4.3% for the FRED legs (CNY, TWD). The longest stale run is 3 weekdays, so no leg has a gap above 5 weekdays. Runs of identical levels that `leg_stale` does not flag are in remaining item 5.

**Check 5 detail.** `asset_returns` equals `leg_returns[long] − leg_returns[short]` exactly (maximum difference 0). `leg_returns` equals L_t / L_{t−1} − 1 from `leg_levels` exactly. CSV and parquet agree to 1e-16. Every stale day has a return of 0.

**Check 6 detail.** The correlation of the JP leg (JPY terms) with JPY v USD is −0.07 daily and +0.06 weekly over the full sample, and +0.09 daily and +0.23 weekly in 2022. The unconverted XJSE.DE price correlates +0.86 weekly with JPY v EUR, so the conversion removes the currency as intended.

**Check 7 detail.** The rerun took about 20 seconds, used the same source for all 58 legs and had no failed download. Every `close` value is identical, and so are all FX legs, DXY, the EM basket and the JP leg. `adj_close` differs by up to 1.2e-6 (relative) on 30 dividend-adjusted series, because Yahoo recomputes its adjustment factors on each call. Asset returns therefore differ by up to 1.9e-6 (99th percentile 7e-7) for 30 of 55 assets. That misses the 1e-10 target but is immaterial: the largest change in annualised vol is 2e-7.

### Fixes made

1. **Bad-tick threshold lowered from 8 to 6 robust sigmas** (`--fx-bad-tick-sigma`, default now 6). At 8, confirmed bad ticks survived in the delivered data. The change drops 10 more prints in legs that use Yahoo:
   - COP 2015-08-10 (+10.1% then −10.1%);
   - NOK 2018-12-31 (+4.1% then −3.3%) and CAD 2020-12-31 (+3.3% then −2.9%), both year-end bars;
   - IDR 2025-06-11 and 2025-06-16 (about ±3% each);
   - THB 2016-01-08, 2016-03-04, 2016-07-01, 2016-11-25 and 2016-12-16 (±2.0–2.6%, all Fridays).

   Where a FRED series exists, FRED moved less than 0.8% on each of those days. COP and IDR have no FRED series, but each print reverses a 3–10% move in full the next day. No genuine move was removed. Six assets change: annualised vol of COP v USD 18.02% → 17.50%, THB 7.35% → 7.03%, CAD 7.30% → 7.18%, NOK 12.16% → 12.06%, IDR 10.00% → 9.85%, EM v USD 6.34% → 6.31%. Sources and statuses are unchanged.
2. **The FX quality gate now uses the unrounded correlation.** Before, the weekly correlation was rounded to 3 decimals and then compared with 0.8, so a value from 0.7995 to 0.8 would have passed. No outcome changes in this run.

### Remaining items

1. **TWD gate margin** (resolved 2026-09-29: TWD pinned to FRED in `legs.csv`). With the stricter cleaning, Yahoo TWD's weekly correlation with FRED rises from 0.762 to 0.7993, just under 0.8. A Yahoo revision can switch TWD back to Yahoo, whose 2015–2016 data is poor (weekly correlation with FRED 0.54–0.57, vol 2.2–2.3x FRED's). I suggest making FRED DEXTAUS the TWD proxy in `legs.csv`, but that is a registry decision for the owner. Until then, check `summary.fallback_used` in `manifest.json` after each rerun.
2. **Friday FX levels are Sunday-evening snapshots** (TBC). After re-dating, the Friday level is Yahoo's Monday 00:00 London bar, so weekend gaps and thin Sunday quotes land on Friday. This also affects Friday-to-Friday weekly returns. Example: EUR −2.0% on 2015-06-26 and +2.3% on 2015-06-29 is the Greek referendum weekend gap. 8 of the 12 one-day reversals above 1.5% that remain in Yahoo FX legs fall on a Friday.
3. **Reversals below the new threshold** [VERIFY]:
   - BRL 2020-05-08, +6.4% then −5.9% (Yahoo 5.48 vs FRED 5.75 BRL per USD);
   - THB Fridays in 2015–2016 at about ±2% (2015-06-26, 2015-12-11, 2016-01-15, 2016-09-23);
   - IDR 2020-01-27, 2020-02-27, 2020-05-01 and 2020-06-11 (±2.5–3.9%);
   - year-end bars at about ±1.7–2% on CAD 2018-12-31, ZAR 2018-12-31, SEK 2019-12-31 and CHF 2020-12-31.

   A check against FRED would catch the pairs that have FRED; COP, CLP, IDR and CZK have none (TBD).
4. **CLP 2022-07-20 (+13.3%)** is a stale-then-jump: Yahoo is flat around 1050 for three days, then moves in one day. The weekly total looks plausible; the daily path does not [VERIFY].
5. **Identical levels that `leg_stale` does not flag**: IITB.MI 2022-09-13 to 2022-09-20 (6 zero returns). FRED repeats the CNY rate over Chinese holidays (5–6 days at each Lunar New Year and Golden Week), and FRED quotes TWD to 2 decimals, so 10% of CNY days and 12% of TWD days have a zero return.
6. **JP leg levels** [VERIFY]: in JPY the leg returns −8.1% in 2022, −4.6% in 2024 and −7.3% in 2025. Compare one year with the JPY total return of the fund's index to confirm the conversion and Yahoo's adjusted close.
7. **Spread vol above the long leg**: EUR IG v Global Credit is 5.2% against 4.3% for the EUR IG leg. CORP.L is unhedged USD while IEAC.L is EUR-local, so the spread carries EUR/USD; this follows the recorded hedging choice (Section 6, item 11). For JP 7-10y v Global Duration, see Section 6, item 4.
8. **Year-end 2025 FX level** is still carried forward from 2025-12-30 (Section 6, item 10).
