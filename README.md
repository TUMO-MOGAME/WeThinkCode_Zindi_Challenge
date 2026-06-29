# Gauteng Service-Delivery Hotspots Challenge

Multi-label classification: predict **5 binary service-gap labels** per Gauteng household
(water, sanitation, refuse, energy, education) from the GCRO Quality of Life Survey.

**Official metric — weighted-average F1:**
```
Score = 0.3304*F1(Water) + 0.2329*F1(Refuse) + 0.2119*F1(Education)
      + 0.1200*F1(Energy) + 0.1048*F1(Sanitation)
```

## Quick start
```bash
python -m venv .venv
.venv\Scripts\activate              # Windows
pip install -r requirements.txt

# 1. raw CSVs -> cleaned interim parquet
python -m src.data.make_dataset --config config/config.yaml

# 2. cross-validate the active model (config/model.yaml -> `active`)
python -m src.cv.run_cv --config config/config.yaml --model-config config/model.yaml

# 3. (optional) tune the active model with Optuna
python -m src.tune --config config/config.yaml --model-config config/model.yaml --trials 30

# 4. fit on train, predict, build a submission
python -m src.train   --config config/config.yaml --model-config config/model.yaml --name exp_001_logreg
python -m src.predict --config config/config.yaml --name exp_001_logreg
python -m src.submit  --config config/config.yaml --name exp_001_logreg
```

## Layout
```
config/        YAML: paths, metric weights, CV strategies (config.yaml), model zoo (model.yaml), features
data/raw       Original Zindi CSVs (gitignored)   data/interim  cleaned   data/processed  encoded
src/data       load + clean + validate
src/features   encoders / feature builders
src/cv         splitters (multilabel-stratified + alternatives), weighted-F1 metric, OOF driver
src/models     registry/zoo, model wrappers, ensemble
src/postprocess per-label F1-max threshold tuning
src/utils      seed / io / logging
src/{train,predict,submit,tune}.py   CLI entry points
experiments/   per-experiment configs, fold scores, OOF, thresholds, scores
models/        saved models + metadata        submissions/  numbered CSVs + submission_log.csv
notebooks/     EDA + the documented Colab submission notebook
```

## Strategy notes
- Only **25 submissions total** -> trust local `val.csv` weighted-F1, not the leaderboard.
- `val.csv` is for **per-label threshold tuning + local scoring only** (never trained on).
- Data has ~825 unique feature profiles -> heavy label noise -> regularise hard; the win is
  threshold tuning, not model capacity. `test.pop_group` has a category unseen in train, so all
  encoding uses `handle_unknown='ignore'`.
