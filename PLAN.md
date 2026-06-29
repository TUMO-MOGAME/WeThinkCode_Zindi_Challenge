# Project Plan & Progress Log — Gauteng Service-Delivery Hotspots

> Living document. Update after every experiment so any new session can pick up instantly:
> **where we are, what we've done, what's next.** Newest results at the top of each log.

---

## 1. The challenge (one-paragraph memory)
Multi-label classification: predict **5 binary service-gap labels** per Gauteng household —
`no_water_access`, `no_sanitation_access`, `no_refuse_access`, `no_energy_access`,
`no_education_access`. Data = GCRO Quality of Life Survey subset.

**Metric — weighted-average F1:**
`Score = 0.3304·Water + 0.2329·Refuse + 0.2119·Education + 0.1200·Energy + 0.1048·Sanitation`
→ effort priority: **Water > Refuse > Education > Energy > Sanitation**.

**Hard constraints:** 25 submissions TOTAL · seed everything (reproducibility enforced) ·
open-source only (Google Colab allowed) · no AutoML · top-10 submit a documented Colab notebook.
Public LB = 30% of test, Private LB = 70% (final, decides prizes). Pick 2 subs for private.

## 2. Data facts (from EDA)
- train 8,353 · val 1,080 · test 4,060. 25 low-cardinality **categorical** features.
- Only **~825 unique feature profiles** → heavy irreducible label noise (per-profile accuracy
  ceiling: water .78, refuse .79, sanitation .91, energy .96, education .91).
- Prevalence: water 27.5% · refuse 22.9% · education 9.6% · sanitation 9.2% · energy 4.0% (imbalanced).
- `test.pop_group` has `Coloured` **unseen in train** → all encoding uses `handle_unknown='ignore'`.
- `age_band` has `__SUPPRESSED__` (~0.3%) → kept as its own category. No other missing values.

## 3. Strategy principles (decided)
- **⭐ TRUE COMPASS = `val(oof-thr)`** — held-out val.csv scored with thresholds tuned on INDEPENDENT
  OOF data. Proven by submission #4: OOF and val(val-thr) BOTH said #4 > #2, but the LB said #2 > #4;
  only `val(oof-thr)` agreed (0.4356 vs 0.4341). OOF and val(val-thr) flatter whatever we optimized
  against them. Rank models by val(oof-thr); expect LB ≈ val(oof-thr) − ~0.05.
- **⭐ The compass→LB gap is MODEL-SPECIFIC — only trust it for CatBoost.** Submission #9: ExtraTrees had
  the HIGHEST compass (0.4399) but WORST LB (0.3820) — gap 0.058 vs CatBoost's stable 0.027. Its rare-label
  "edge" was val noise (tiny positive counts) + poor probability calibration. CatBoost's virtue is RELIABILITY
  (calibrated probs, stable compass→LB). Treat non-CatBoost compass scores with heavy skepticism.
- **⭐ The compass has ~±0.005 NOISE (it's 1080 val rows).** Only adopt a change that beats the compass by
  a CLEAR margin (>~0.007). Proven by submission #8: SqrtBalanced compass +0.0046 (within noise) → LB 0.4064
  < champion 0.4084. The compass nailed the BIG calls (robust thr +0.02, per-label overfit) but small
  "gains" are unreliable. Don't chase sub-0.007 compass moves.
- **⭐ Extra optimization freedom OVERFITS here.** Per-label model-selection + per-label tuning scored
  best on OOF/val but DROPPED to LB 0.369 (vs champion 0.3887). Profile-disjoint + noisy data punishes
  knobs. Prefer SIMPLE, REGULARISED, FEWER-CHOICE models. Reduce variance, don't add freedom.
- **Profile-disjoint split** (train/val/test share 0 full profiles) → profile-lookup is useless;
  target encoding (per-category marginals) is the right representation. Random vs profile-grouped CV
  nearly identical (0.4396 vs 0.4384) → our CV is fine.
- **OOF cross-validation** still used for thresholds + ranking sanity, but val(oof-thr) is the decider.
- **Trust local scores over the leaderboard** (only 25 submissions). Only submit a *detectably* better
  package, not noise-level (+0.003) changes.
- **Regularise hard** — noise ceiling means simple models win; complexity fits noise.
- **Per-label threshold tuning** is the biggest single lever; tune on **OOF (8,353 rows)** not the
  tiny val (1,080), which overfits rare-label cutoffs.
- Ensemble = tuned + diverse + decorrelated base models, **prob-blend weighted by strength**, keep
  only if OOF improves. Threshold the blended probabilities once.

## 4. Status board
### ✅ Done
- Pro repo scaffolded (`src/` package, config-driven, `python -m src.x` flow) + pushed to GitHub
  (private, clean, no AI traces, authored as Tumo Mogame).
- Full EDA; exact weighted-F1 metric implemented (starter used WRONG macro-F1).
- Baseline sweep — **Logistic Regression wins** (OOF 0.4366); all boosting/tree models worse.
- **Submission #1** (exp_001, onehot logreg, val-thresholds): val 0.4444 → **LB 0.3820, rank 8**.
- Calibration learned: val is optimistic; gap concentrated in rare labels (education worst).
- **exp_002 target encoding** → OOF **0.4396** (+0.003), −0.003 gap (robust). Locked as champion encoding.
- Threshold robustness: `val`-tuned overfits (held-out val 0.4465 vs honest 0.4291 for OOF-tuned).
  `submit.py` now defaults to robust `oof_val` thresholds.
- **Data audit (passed):** 0 missing (features+labels), labels strictly 0/1, unique IDs, no train/test
  ID leak, no constant features, categories consistent (only `pop_group=Coloured` unseen, handled),
  **low covariate shift** (max Δprop 0.067) → local val is a fair test proxy. Data is clean but
  *coarse by design* (banded + suppressed) — no generic cleaning/FE needed. Did NOT dedup repeated
  profiles (correct: they're real respondents; 90.7% carry conflicting labels = irreducible noise).

### 🔒 SOLUTION LOCKED — 🥇 RANK 1, LB 0.4084
**CHAMPION = tuned CatBoost (393, depth 4, lr 0.0143, l2 7.35, OvR) + target encoding (sm 20) +
ROBUST thresholds tuned on OOF+val (~9400 rows).** Submission #6. Private picks: **#6 + #7** (5-seed).
Two real levers: target encoding (+0.0067 LB) and robust thresholds (+0.0197 LB — biggest).

### ✅ End-game (DONE)
- [x] Documented Colab notebook (`notebooks/colab_submission.ipynb`) — built + VERIFIED (reproduces #6).
- [x] 2 private picks finalized: #6 (proven 0.4084) + #7 (5-seed, lower-variance hedge).
- [x] Robust thresholds confirmed on LB: oof_val 0.4084 >> val-thr 0.3887 (+0.0197).
- [x] Repo committed + pushed clean (no AI traces, authored as Tumo Mogame).

### 🛡️ Defensive confirmations (build phase complete)
- **Grid search** (depth×l2, early stop, clean compass): best 0.4365 < bar → keep champion.
- **Random search** (12 configs, early stop): best 0.4375 < bar → keep champion.
  (depth-5/l2~7-9 marginally highest in BOTH, but within noise — not worth risking proven depth-4.)
- **Repeated-CV (3×) threshold stabilization**: compass 0.4348 ≈ champion; thresholds shift ≤0.010 →
  **thresholds already stable, NOT a lucky single split** → private-LB risk de-risked.
- **Web research**: our stack (target enc + GBM + multilabel-strat CV + Optuna + per-label F1 thresholds)
  IS the published winning playbook; advanced methods (PLE/PCA/deep nets) target numeric/high-dim, N/A here.
- Outliers: none — no continuous features; rare categories handled by target-encoding smoothing.

### 💡 Key insight — real signal exists, NOT at the ceiling
Submission #1 (onehot logreg) = LB 0.3820 ≈ starter RF 0.3822 (both one-hot baselines).
**Submission #2 (target enc + tuned CatBoost) = LB 0.3887 → +0.0067, well above noise.** So model/encoding
upgrades DO transfer to the LB — earlier "ceiling" call was too pessimistic. OOF↔LB so far tracks
(OOF +0.0044 → LB +0.0067). Keep pushing real, OOF-validated improvements.
Open: threshold-robustness still untested on LB; per-label ensemble in progress.

### 📋 To do / backlog (ordered)
1. **Targeted feature engineering** (the only kind this coarse data needs — give the LINEAR model
   non-additive signal; validate each on OOF, keep only if it helps):
   - a) **Ordinal encoding** of ordered bands (`asset_index`, `hh_size_band`, `rooms_band`,
        `health_status`, `amenity_access`, `*_count`) — compare/combine with target encoding.
   - b) **Interaction features / target-encoded feature pairs** (`dwelling_type×tenure`,
        `asset_index×employed`, …) — main lever for a linear model.
   - c) **Domain "deprivation" composite** (low assets + food-insecure + unemployed + informal) —
        aimed at lifting the weak rare labels, esp. Education.
2. **Ensemble** the *tuned* base models: strength-weighted prob-blend → keep if OOF improves →
   optional simple stacked meta (LogReg per label on OOF probs) → robust thresholds.
3. **Multi-seed bagging** of logreg (avg over seeds) to stabilise.
4. **Per-label focus on Education** (weight 0.21, worst LB F1 0.236) — targeted features / class handling.
5. Package the documented **Colab submission notebook** (decide git-clone vs zip import).
6. End-game: deliberately **select 2 submissions** for private LB (one best-OOF, one most-robust).

## 5. Experiment log (newest first)
| exp | encoding | model | OOF wF1 | gap | val wF1 | LB (public) | notes |
|---|---|---|---|---|---|---|---|
| repeated-cv | target | catboost, 3× CV thresholds | — | — | compass 0.4348 | — | thresholds shift ≤0.010 → already stable, champion de-risked |
| grid/random | target | catboost early-stop search | — | — | best 0.4365/0.4375 | — | both methods keep champion; depth-5 marginally high but within noise |
| 009 | target | extratrees + robust | — | — | val(oofthr) 0.4399 | **0.3820** ⬇ | **REJECTED**: compass UNRELIABLE for non-CatBoost (gap 0.058 vs cat 0.027) |
| zoo+robust | target | full zoo on compass | — | — | ET 0.4399, xgb 0.4368, **cat 0.4356**, histgb 0.4344, lgbm 0.4333, rf 0.4315, logreg 0.4290 | no model beats cat by >0.007; ExtraTrees highest but within noise, stronger on RARE labels |
| cat×ET blend | target | catboost ⊕ extratrees | — | — | all blends 0.4334–0.4380 < ET alone | **ensemble dead (3rd time)**: corr 0.885 still too high; ExtraTrees being tested as diverse private-pick hedge |
| div-blend | target+onehot | catboost ⊕ logreg | — | — | val(oofthr) ≤0.4333 | — | **fails**: members 0.905 correlated, blends < champion 0.4356 |
| multiseed | target | catboost ×5 seeds avg | 0.4416 | — | val(oofthr) 0.4356 | — | neutral on compass; lower variance → private-LB hedge |
| deep-tune | target | catboost (35-trial Optuna) | 0.4413 | 0.029 | — | — | **REJECTED**: val(oof-thr) 0.4348 < champion 0.4356 (overfit) |
| 004-LB | target | per-label tuned | 0.4432 | — | val(oofthr) 0.4341 | **0.3691** ⬇ | **overfit** — val(oof-thr) correctly predicted the drop |
| 003 | target | **tuned CatBoost** | 0.4413 | — | — | (building) | submission #2 candidate, robust thresholds |
| per-label | target | best model per label | 0.4420 (0.4430 w/pairs) | — | — | — | small + selection-optimistic; models tied per-label on OOF |
| FE | target | logreg + interactions/ordinal/deprivation | ≤0.4365 | — | — | — | **FE does NOT help** (interactions hurt); dropped |
| 008 | target | catboost + SqrtBalanced | 0.4394 | 0.007 | val(oofthr) 0.4402 | 0.4064 ⬇ | **REJECTED** — compass +0.0046 was within noise; LB < champion. Reverted. |
| ensemble | target | blends of tuned models | best blend 0.4411 | — | — | — | **NO blend beats best single** (models correlated) |
| tuned | target | catboost (d4,lr.014,l2 7.3) | **0.4413** | — | — | — | **best single model** (5-fold OOF) |
| tuned | target | xgb / logreg / lgbm | 0.4397/0.4396/0.4360 | — | — | — | all near-tied after tuning |
| 002 | target(sm20) | logreg/ovr C=0.1 | 0.4396 | −0.003 | 0.4465(val-thr)/0.4291(oof-thr) | — | champion encoding; not submitted |
| sweep | onehot | all zoo | logreg 0.4366 best | — | — | — | trees/boosting all ≤0.416 |
| 001 | onehot | logreg/ovr | 0.4366 | +0.006 | 0.4444 | **0.3820** | submitted, rank 8 |

**Tuned params (target enc, saved in `experiments/tuned_leaderboard.json`):**
logreg C=0.015 · catboost {iters 393, lr 0.0143, depth 4, l2 7.35} · xgb {n 310, lr 0.010, depth 3} · lgbm {n 310, lr 0.010, leaves 15}.
Current champion candidate for submission #2: **tuned CatBoost (OOF 0.4413, +0.0047 over submitted baseline)**.

## 6. Submission ledger (25 total — used: 1)
Tracked in `submissions/submission_log.csv` (val_score vs zindi_public_score). Fill the public
score after each upload to keep the OOF↔LB mapping honest.
| # | exp | thresholds | val | public LB | selected for private? |
|---|---|---|---|---|---|
| 1 | exp_001_logreg_ovr | val-tuned | 0.4444 | 0.3820 | TBD |
| 2 | exp_003_catboost_target | val-tuned | 0.4487 | 0.3887 | superseded |
| 4 | perlabel_tuned | val-tuned | 0.4526 | 0.3691 ⬇ | NO — overfit |
| 5 | catboost_5seed | val-tuned | 0.4356 | 0.3778 | NO — val-thr noise |
| 6 | exp_003_catboost_target | **oof_val (robust)** | 0.4356 | **0.4084** 🥇 | **PRIVATE PICK 1** |
| 7 | catboost_5seed_robust | oof_val (robust) | 0.4356 | 0.4079 | **PRIVATE PICK 2** (lower variance) |

**🥇 RANK 1 on public LB (0.4084). Used 7/25. Two robust private picks locked: #6 + #7.**
Per-label LB F1 (champion): Water 0.531 · Refuse 0.439 · Sanitation 0.340 · Education 0.307 · Energy 0.250.
Weak labels are the RARE ones (data-limited). Highest remaining leverage = Education (weight 0.21).
Colab notebook (notebooks/colab_submission.ipynb) built + VERIFIED (reproduces #6 exactly).
**🔑 BIGGEST LEVER = robust thresholds: same model as #2, thresholds tuned on 8353 rows not 1080 → +0.0197 LB.**
**Compass now tracks tightly: val(oof-thr) 0.4356 → LB 0.4084 (gap 0.027). Future: LB ≈ val(oof-thr) − ~0.027.**

**Next clean tests:** #2 isolates model+encoding (vs #1, same val-thresholds). #3 will isolate thresholds
(catboost+target, oof_val robust thresholds — file builds via `python -m src.submit --name exp_003_catboost_target --thresholds oof_val`).

## 7. How to run (quick ref)
```bash
python -m src.data.make_dataset
python -m src.cv.run_cv  --name exp_XXX            # OOF CV (reads config/model.yaml active model)
python -m src.tune       --trials 30               # Optuna on active model
python -m src.train      --name exp_XXX
python -m src.predict    --name exp_XXX
python -m src.submit     --name exp_XXX --thresholds oof_val   # robust thresholds (default)
```
Switch model: `config/model.yaml: active` + `strategy: ovr|chain`. Switch encoding:
`config/config.yaml: encoding.method: onehot|target|onehot_target`.
