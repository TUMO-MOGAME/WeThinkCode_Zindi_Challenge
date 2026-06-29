# Solution Documentation — Mapping Service-Delivery Gaps in Gauteng
**Author:** Tumo Olorato Mogame
**Public leaderboard score:** 0.4084 (weighted-average F1) · **Private leaderboard score:** revealed at challenge close

This document follows Zindi's recommended documentation structure: overview, architecture, ETL,
data modeling, inference, run time, performance metrics, error handling & logging, and maintenance.

---

## 1. Overview and objectives

**Purpose.** Predict, for each household in Gauteng, five binary **service-gap** labels —
`no_water_access`, `no_sanitation_access`, `no_refuse_access`, `no_energy_access`,
`no_education_access` — from 25 categorical survey features. This is a **multi-label classification**
problem; the gaps tend to co-occur. The output helps direct government resources to the households
that need them most.

**Metric (objective optimised).** A weighted average of per-label F1:
```
Score = 0.3304·F1(Water) + 0.2329·F1(Refuse) + 0.2119·F1(Education) + 0.1200·F1(Energy) + 0.1048·F1(Sanitation)
```

**Expected outcome.** A reproducible model that, given the 25 features for unseen households,
outputs the five 0/1 labels, scored by the weighted F1 above. Achieved: **0.4084 public** (rank 1 at
time of writing).

**Solution in one line.** Leak-safe multi-label **target encoding** → regularised **CatBoost** per
label → **per-label decision thresholds tuned on a large out-of-fold + validation sample.**

---

## 2. Architecture diagram (data flow)

```
                ┌──────────────────────────────────────────────────────────────┐
                │                        RAW DATA (CSV)                          │
                │            train.csv · val.csv · test.csv · SampleSubmission   │
                └───────────────┬──────────────────────────────────────────────┘
                                │  EXTRACT (src/data/make_dataset.py)
                                ▼
                ┌──────────────────────────────────────────────────────────────┐
                │  TRANSFORM 1 — cleaning & validation (src/data/{make_dataset,  │
                │  validate}.py): cast to string, keep __SUPPRESSED__, schema    │
                │  & unseen-category checks                                      │
                └───────────────┬──────────────────────────────────────────────┘
                                │  LOAD -> interim parquet (data/interim/)
                                ▼
                ┌──────────────────────────────────────────────────────────────┐
                │  TRANSFORM 2 — leak-safe multi-label TARGET ENCODING           │
                │  (src/features/target_encoding.py), fit inside each CV fold    │
                └───────────────┬──────────────────────────────────────────────┘
                                ▼
   ┌───────────────────────────────────────────┐     ┌──────────────────────────────────┐
   │  MODELING (src/cv, src/models)            │     │  THRESHOLDS (src/postprocess)     │
   │  CatBoost (OvR) + MultilabelStratified    │ --> │  per-label F1-max cutoffs tuned   │
   │  5-fold OOF; metric = weighted-F1         │ OOF │  on OOF(train)+val (~9,400 rows)  │
   └───────────────────────┬───────────────────┘     └───────────────┬──────────────────┘
                           │  fit on full train                       │
                           ▼                                          ▼
                ┌──────────────────────────────────────────────────────────────┐
                │  INFERENCE (src/predict.py, src/submit.py)                     │
                │  predict_proba(test) → apply thresholds → binary labels        │
                │  → validate vs SampleSubmission → submission.csv               │
                └──────────────────────────────────────────────────────────────┘
```

---

## 3. ETL process

**Extract.** Data source = the competition CSV files (`train.csv` 8,353 rows with labels,
`val.csv` 1,080 labelled rows, `test.csv` 4,060 unlabelled rows, plus `SampleSubmission.csv`).
Volume is small (~3 MB total); extraction is a one-time read with pandas. No external sources are used.

**Transform.**
- *Cleaning / validation* (`src/data/make_dataset.py`, `src/data/validate.py`): every feature is cast
  to **string** so all 25 features are treated as categories; the privacy token
  `age_band == "__SUPPRESSED__"` is kept as its own category. A schema check confirms feature columns
  and submission columns; an unseen-category scan flags `pop_group="Coloured"` (present in test, not
  train) so encoders absorb it via `handle_unknown='ignore'`. A data audit verified no missing values,
  labels strictly 0/1, unique IDs, no train/test ID leakage, and low covariate shift. There are no
  continuous features, so no outlier removal is required.
- *Feature transform* (`src/features/target_encoding.py`): leak-safe multi-label **target encoding** —
  each `(feature, label)` category is replaced by the smoothed mean of that label (smoothing = 20),
  computed **out-of-fold** (internal cross-fitting) and fit only on training folds. 25 features × 5
  labels → 125 numeric columns.

**Load.** Cleaned data is written to **interim parquet** (`data/interim/`) for fast reuse; out-of-fold
predictions, thresholds and per-run scores are persisted under `experiments/<run>/`; the fitted model
is saved with `joblib`. The final binary predictions are written to `submissions/submission.csv`.

---

## 4. Data modeling

**Assumptions / foundation.** The train/val/test split is **profile-disjoint** (no full 25-feature
profile is shared across splits — verified), so a model cannot memorise households; it must generalise
through **per-category statistics**. Target encoding is the representation that captures exactly this.

**Feature selection / engineering.** All 25 provided categorical features are used, via target
encoding (Section 3). Manual feature engineering (interactions, ordinal encodings, a deprivation
composite) was tested and *reduced* cross-validation performance, so none is used. No normalization is
required (target-encoded values are already on a 0–1 scale; CatBoost is scale-invariant).

**Model / algorithm.** CatBoost (`CatBoostClassifier`, gradient-boosted decision trees), trained
**one-vs-rest** (`MultiOutputClassifier` — one CatBoost per label). Chosen after benchmarking the full
zoo (LogReg, RandomForest, ExtraTrees, HistGB, XGBoost, LightGBM, CatBoost): CatBoost was both
highest-scoring and the most **reliable / well-calibrated** (its local-vs-leaderboard gap was the
smallest and most stable).

**Hyper-parameters.** `iterations=393, depth=4, learning_rate=0.0143, l2_leaf_reg=7.35` — found with
Optuna and independently confirmed by grid and random search; early stopping selected ~363 iterations,
confirming the choice. Settings are deliberately strong on regularisation (train↔CV gap ≈ 0.007).

**Training process.** Fit one CatBoost per label on target-encoded features, on `train.csv` only.

**Validation.** 5-fold **MultilabelStratifiedKFold** out-of-fold (OOF) cross-validation. The selection
metric is the exact competition **weighted-F1**. The honest "compass" used to accept/reject every
change was the **validation score computed with thresholds tuned on independent OOF data** — this
tracked the leaderboard closely (e.g. compass 0.4356 → public 0.4084).

**Decision thresholds (key step).** For each label, the F1-maximising cutoff is chosen by grid search
over `[0.05, 0.95]`. Crucially, thresholds are tuned on the **OOF(train) + val sample (~9,400 rows)**,
not the small val set alone — this stabilises the rare-label cutoffs and improved the leaderboard from
0.389 to 0.408 (the single biggest gain). Thresholds were verified stable across repeated CV splits.

---

## 5. Inference

**How new data is scored.** Given new household feature rows (e.g. `test.csv`):
1. Cast features to string (same cleaning as training).
2. Apply the **fitted** target encoder (full-train maps; unseen categories fall back to the global
   mean) → 125 numeric features.
3. `predict_proba` with the trained CatBoost models → a probability per label.
4. Apply the saved per-label thresholds → binary 0/1 labels.
5. Align to `SampleSubmission.csv` and validate the format → `submission.csv`.

**Infrastructure.** Runs locally or in Google Colab on **CPU** — no GPU, no external services. The
self-contained notebook `notebooks/colab_submission.ipynb` performs the full train → predict → submit
flow; the `src/` package exposes the same via `python -m src.{train,predict,submit}`.

**Versioning / retraining.** Each run is parameterised by `config/` (model + encoding + CV) and a fixed
seed (42), so any version is fully reproducible. To retrain on new data, replace the CSVs and re-run
the pipeline; the saved model + thresholds (`models/`, `experiments/<run>/thresholds.json`) define a
deployable version. No online/streaming inference is required for this challenge (batch scoring of
`test.csv`).

---

## 6. Run time

| Stage | Approx. time (CPU) |
|---|---|
| Dependency install | ~1–2 min |
| Data extract + clean | < 5 s |
| 5-fold OOF cross-validation (CatBoost) | ~6–8 min |
| Final fit on full train | ~1–1.5 min |
| Predict val + test, threshold tuning, write submission | < 10 s |
| **Total end-to-end** | **~8–12 minutes** |

Hardware: standard CPU (Google Colab CPU runtime / local CPU). No GPU.

---

## 7. Performance metrics

- **Public leaderboard:** **0.4084** (weighted-average F1) — rank 1 at submission time.
- **Private leaderboard:** revealed at challenge close (the private 70% of the test set).
- **Per-label F1 (public):** Water 0.531 · Refuse 0.439 · Sanitation 0.340 · Education 0.307 · Energy 0.250.
- **Local validation metric used for all decisions:** weighted-F1 on held-out val with OOF-tuned
  thresholds ("compass") — observed gap to the public LB ≈ 0.027, stable.

**Progression (what moved the score):**

| Stage | Public LB |
|---|---|
| LogReg + one-hot baseline | 0.382 |
| CatBoost + target encoding (val thresholds) | 0.389 |
| **CatBoost + target encoding + robust thresholds (final)** | **0.408** |

Two decisive levers: target encoding (+0.007) and robust thresholds (+0.020).

---

## 8. Error handling and logging

- **Schema & input validation** (`src/data/validate.py`): asserts feature columns are present and the
  submission columns/format match `SampleSubmission.csv`; logs any categories present in val/test but
  absent from train (handled, not fatal); reports missing-value counts.
- **Unseen categories** never raise — encoders use `handle_unknown='ignore'` / global-mean fallback.
- **Submission guards** (`src/submit.py`): assertions that there are no missing IDs vs the sample
  submission, that column order matches exactly, and that every predicted value is in `{0, 1}` — a
  malformed submission fails fast rather than being uploaded.
- **Logging** (`src/utils/logging.py`): a console logger reports per-fold scores, the train↔CV gap,
  chosen thresholds, and output paths, so a reviewer can see exactly what happened at each step.
- **Reproducibility guard** (`src/utils/seed.py`): a single seed is set for Python, NumPy and the
  models; re-running yields the same result.

---

## 9. Maintenance and monitoring

- **Retraining.** Drop new/updated CSVs into `data/raw/` and re-run the pipeline; the config-driven
  design means no code changes are needed to refresh the model.
- **Monitoring.** If deployed, track per-label F1 and the predicted positive-rate per label against the
  historical survey rates — a large drift in predicted rates is the earliest sign that thresholds need
  re-tuning (the most sensitive component for rare labels).
- **Scaling.** The data is small and CPU-only; the pipeline scales linearly and would comfortably
  handle far larger surveys. The heaviest step (5-fold CV) is embarrassingly parallel across folds.
- **Lifecycle / versioning.** Every run is fully specified by `config/` + seed and saves its model,
  thresholds and scores under `experiments/<run>/` and `models/`, so versions are auditable and any
  past result can be reproduced exactly.

---

## 10. Reproducibility & environment

- **Seed:** 42 (Python, NumPy, CatBoost). Re-running reproduces the submission.
- **Dependencies:** see `requirements.txt` (open-source only; pinned versions). No AutoML, no GPU.
- **Data:** only the provided competition files; **no external datasets**.
- **Collaboration:** solo work; no private code sharing.
- **Run order:** `make_dataset → run_cv → train → predict → submit` (CLI), or run
  `notebooks/colab_submission.ipynb` top-to-bottom.

---

## 11. What was explored and rejected (design decisions)

Each was measured on the honest compass and did **not** improve generalisation — recorded for transparency:

| Idea | Outcome |
|---|---|
| Full model zoo (LogReg, RF, ExtraTrees, HistGB, XGBoost, LightGBM) | tie/worse than CatBoost; CatBoost most reliable |
| Manual feature engineering (interactions, ordinal, deprivation index) | reduced CV — added variance, no signal |
| Per-label model selection + per-label tuning | overfit — leaderboard dropped 0.389 → 0.369 |
| Deeper Optuna / grid / random search, early stopping | no clear gain; deeper search overfit |
| Model ensembles / probability blends | no gain — diverse models still ~0.9 correlated |
| Class-weighting (Balanced / SqrtBalanced) | within noise; did not transfer to the leaderboard |
| Profile-lookup model | inapplicable — 0% of test profiles appear in train |
| Probability calibration | irrelevant — F1 with a tuned threshold is invariant to monotone rescaling |

---

## 12. Key takeaways

On small, coarse, noisy data the winning approach is **discipline, not complexity**: a simple
regularised model, an honest validation compass, and well-chosen decision thresholds, while rejecting
every change that does not demonstrably improve out-of-sample performance. The only learnable signal is
the per-category gap rate, which target encoding captures and which all reasonable models extract
similarly — so robustness (CatBoost + stable thresholds), not model novelty, decides the result.
