# Solution Documentation — Mapping Service-Delivery Gaps in Gauteng
**Author:** Tumo Olorato Mogame
**Final public leaderboard score:** 0.4084 (weighted-average F1)

---

## 1. Summary

The task is **multi-label classification**: for each Gauteng household, predict five binary
service-gap labels — `no_water_access`, `no_sanitation_access`, `no_refuse_access`,
`no_energy_access`, `no_education_access` — from 25 categorical survey features.

Submissions are scored with a **weighted average of per-label F1**:

```
Score = 0.3304·F1(Water) + 0.2329·F1(Refuse) + 0.2119·F1(Education)
      + 0.1200·F1(Energy) + 0.1048·F1(Sanitation)
```

The final solution is deliberately **simple and heavily regularised**, because the data is small,
coarse (banded categories), and noisy. It has three components:

1. **Leak-safe multi-label target encoding** of the categorical features.
2. A regularised **CatBoost** classifier per label (one-vs-rest).
3. **Per-label decision thresholds tuned on a large out-of-fold + validation sample** — the single
   highest-impact step.

Every more elaborate idea (per-label tuning, deeper hyper-parameter search, model ensembles,
manual feature engineering) was tested against an honest cross-validation compass and **rejected
because it did not improve generalisation** — on this dataset, added model freedom causes overfitting.

---

## 2. Repository structure

```
config/        YAML configuration (paths, metric weights, CV, model, encoding)
data/raw/      Original competition CSVs (train, val, test, SampleSubmission, data_dictionary)
src/
  data/        loading, cleaning, schema validation
  features/    target encoding + feature selection
  cv/          cross-validation splitters, the exact weighted-F1 metric, OOF driver
  models/      model registry/zoo, pipeline assembly, ensembling
  postprocess/ per-label F1 threshold tuning
  utils/       seeding, IO, logging
  train.py / predict.py / submit.py / tune.py     CLI entry points
notebooks/
  colab_submission.ipynb     self-contained, documented notebook that reproduces the solution
experiments/   per-run CV scores, OOF predictions, thresholds (generated)
submissions/   generated submission CSVs + submission log
```

---

## 3. Environment & dependencies

Open-source Python only. No AutoML, no paid services, no GPU.

```
numpy, pandas, pyarrow, scikit-learn, pyyaml, joblib
catboost            (gradient-boosted decision trees)
iterative-stratification   (MultilabelStratifiedKFold)
optuna              (hyper-parameter search, development only)
matplotlib, seaborn (EDA only)
```

Install: `pip install -r requirements.txt` (or the install cell in the notebook).

---

## 4. How to reproduce

**Easiest (single notebook):** open `notebooks/colab_submission.ipynb` in Google Colab, upload the
five challenge CSVs, and run all cells. It regenerates `submission.csv` (the 0.4084 file) end-to-end.

**Via the package (CLI):**
```bash
pip install -r requirements.txt
python -m src.data.make_dataset                                  # clean raw -> interim
python -m src.cv.run_cv   --name champion                        # 5-fold OOF (saves OOF predictions)
python -m src.train       --name champion                        # fit on full train
python -m src.predict     --name champion                        # predict val + test
python -m src.submit      --name champion --thresholds oof_val   # robust thresholds -> submission.csv
```
Model is set in `config/model.yaml` (`active: catboost`); encoding in `config/config.yaml`
(`encoding.method: target`).

---

## 5. Data

- `train.csv` — 8,353 rows, features + 5 labels. Used to **fit** the model.
- `val.csv` — 1,080 rows, labelled. Used **only** to help set decision thresholds and to validate
  locally. **The model is never trained on val.**
- `test.csv` — 4,060 rows, features only. Predictions submitted.

All 25 features are **categorical** (low-cardinality bands such as `asset_index = 3-4`,
`dwelling_type = Informal`), drawn from housing, demographics, income, employment, household
composition, amenity access, health and governance themes.

---

## 6. Data processing & cleaning

A data audit confirmed the data is clean: **no missing values**, labels strictly 0/1, unique IDs,
no train/test ID leakage, and **low covariate shift** between train and test. Processing steps:

- Every feature is cast to **string** so all values are treated as categories.
- `age_band = "__SUPPRESSED__"` (a privacy token, ~0.3% of rows) is kept as its **own category**.
- One category (`pop_group = "Coloured"`) appears in test but not train; all encoders use
  `handle_unknown='ignore'` / a global-mean fallback so unseen categories never break inference.
- There are **no continuous numeric features**, so outlier removal does not apply. The categorical
  analogue (very rare categories) is handled by encoding smoothing (Section 7).
- Repeated feature rows are **not** removed — they are real survey respondents, and the model must
  learn the conditional gap rate for each household profile.

A key structural fact: the features collapse to ~825 unique "profiles", and **no full 25-feature
profile is shared across train, val and test** (verified). A model therefore cannot memorise
profiles; it must generalise through **per-category statistics**, which directly motivates the
encoding below.

---

## 7. Feature engineering — leak-safe multi-label target encoding

Each `(feature, label)` category is replaced by the **smoothed mean of that label** within the
training data:

```
encoded(category) = (count·mean + m·global_mean) / (count + m),   m = smoothing (= 20)
```

This gives the model the predictive signal directly (the gap rate for households of each category),
and the smoothing shrinks rare categories toward the global mean so they are not noisy. Because full
profiles never repeat across splits, these per-category marginals are exactly what transfers to the
test set.

**Leakage control:** encoding is computed **out-of-fold** — internal K-fold cross-fitting inside
`fit_transform`, so a training row never sees its own label — and is fit only on training folds.
Implementation: `src/features/target_encoding.py` (`MultiLabelTargetEncoder`, a scikit-learn
transformer). 25 features × 5 labels → 125 encoded numeric columns.

Manual feature engineering (pairwise interactions, ordinal encodings, a "deprivation" composite) was
tested and **reduced** cross-validation performance, so none is used.

---

## 8. Cross-validation strategy

Five labels cannot be balanced with ordinary `StratifiedKFold`, so validation uses
**MultilabelStratifiedKFold** (5 folds). For each candidate, out-of-fold (OOF) probabilities are
collected on the full training set.

The decision metric throughout is the **exact competition weighted-F1** (`src/cv/evaluate.py`), not
the macro-F1 that a naive baseline reports. The honest model-selection "compass" used is the
**validation score obtained with thresholds tuned on independent OOF data** — this proved to track
the leaderboard closely and was used to accept/reject every change.

---

## 9. Model

- **Algorithm:** CatBoost (`CatBoostClassifier`), gradient-boosted decision trees, chosen because it
  was both the highest-scoring and the most **reliable / well-calibrated** model in a full
  benchmark (LogReg, RandomForest, ExtraTrees, HistGB, XGBoost, LightGBM, CatBoost were all compared).
- **Multi-label strategy:** one-vs-rest (`MultiOutputClassifier`) — one CatBoost per label.
- **Hyper-parameters** (found with Optuna, confirmed independently by grid and random search):
  `iterations=393, depth=4, learning_rate=0.0143, l2_leaf_reg=7.35`. These are deliberately strong
  on regularisation; the train↔CV gap is ~0.007 (no overfitting). Early stopping independently
  selected ~363 iterations, confirming the choice.

---

## 10. Decision-threshold optimisation (the key step)

The labels are imbalanced (Energy ~4%, Education/Sanitation ~9%), so the F1-optimal cutoff is far
from 0.5. For each label independently, the threshold that maximises that label's F1 is chosen by a
grid search over `[0.05, 0.95]`. (Optimising each label separately is optimal because the weighted
score is a sum of independent per-label F1 terms.)

**Critically, thresholds are tuned on the combined out-of-fold (8,353) + validation (1,080) sample
(~9,400 rows), not on the small validation set alone.** Tuning on the tiny val set overfits the
cutoffs for rare labels; using the large OOF+val sample makes them stable. This single change
improved the leaderboard from **0.389 to 0.408** — a larger gain than any model change — and the
resulting thresholds were verified to be stable across repeated cross-validation splits.

---

## 11. Final submission generation

1. Fit the champion CatBoost on all of `train.csv` (target-encoded features).
2. Predict probabilities for `val.csv` and `test.csv`.
3. Tune per-label thresholds on OOF(train) + val.
4. Apply those thresholds to the test probabilities to produce binary labels.
5. Align to `SampleSubmission.csv`, validate the format, and save `submission.csv`.

---

## 12. Reproducibility

- **Seed:** a single global seed (42) is set for Python, NumPy and all models. Re-running reproduces
  the same submission.
- **Hardware:** CPU only — no GPU used.
- **Runtime:** approximately 8–12 minutes end-to-end on a standard CPU (Google Colab), including
  dependency installation; the time is dominated by 5-fold cross-validation and the final fit.
- **No external data** was used — only the provided competition files.
- **No collaboration** outside the team and no private code sharing.

---

## 13. Results

| Stage | Public LB |
|---|---|
| Logistic-regression / one-hot baseline | 0.382 |
| CatBoost + target encoding (val thresholds) | 0.389 |
| **CatBoost + target encoding + robust thresholds (final)** | **0.408** |

The two decisive levers were **target encoding** (+0.007) and **robust thresholds** (+0.020).

Per-label F1 (final): Water 0.531 · Refuse 0.439 · Sanitation 0.340 · Education 0.307 · Energy 0.250.

---

## 14. What was explored and rejected (and why)

Documented for transparency — each was measured on the honest compass and did **not** help:

| Idea | Outcome |
|---|---|
| Full model zoo (LogReg, RF, ExtraTrees, HistGB, XGBoost, LightGBM) | all tie/worse than CatBoost; CatBoost most reliable |
| Manual feature engineering (interactions, ordinal, deprivation index) | reduced CV — added variance, no signal |
| Per-label model selection + per-label tuning | overfit — leaderboard dropped 0.389 → 0.369 |
| Deeper Optuna / grid / random search, early stopping | no clear gain; deeper search overfit |
| Model ensembles / probability blends | no gain — diverse models still ~0.9 correlated |
| Class-weighting (Balanced / SqrtBalanced) | within noise; did not transfer to the leaderboard |
| Profile-lookup model | inapplicable — 0% of test profiles appear in train |
| Probability calibration | irrelevant — F1 with a tuned threshold is invariant to monotone rescaling |

---

## 15. Key takeaways

On small, coarse, noisy data the winning approach is **discipline, not complexity**: a simple
regularised model, an honest validation compass, and well-chosen decision thresholds, while rejecting
every change that does not demonstrably improve out-of-sample performance. The only learnable signal
is the per-category gap rate, which target encoding captures and which all reasonable models extract
similarly — so robustness (CatBoost + stable thresholds), not model novelty, decides the result.
