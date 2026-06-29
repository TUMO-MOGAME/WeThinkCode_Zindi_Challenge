"""Tune per-label thresholds, apply to test, write a numbered, validated submission.

Threshold source (--thresholds):
  val      - tune on val.csv only (the official starter pattern; fragile for rare labels)
  oof      - tune on train out-of-fold probs (8353 rows, far more positives -> stable)
  oof_val  - tune on OOF + val combined (default: most positives, most robust)

`oof`/`oof_val` require `python -m src.cv.run_cv --name <exp>` to have produced
experiments/<exp>/oof/oof_predictions.parquet with the SAME encoding/model as this experiment.
"""
import os, argparse, glob
import numpy as np, pandas as pd
from .utils.io import load_config, ensure_dir, save_json
from .utils.logging import get_logger
from .data.make_dataset import build as build_data
from .cv.evaluate import competition_score
from .postprocess.thresholds import tune_thresholds, apply_thresholds, make_grid

def _threshold_data(cfg, exp_name, source, train, val, labels):
    """Return (proba, y) used to TUNE thresholds, per the chosen source."""
    exp_dir = os.path.join(cfg["paths"]["experiments_dir"], exp_name)
    val_proba = pd.read_parquet(os.path.join(exp_dir, "val_proba.parquet"))[labels].values
    y_val = val[labels].values.astype(int)
    if source == "val":
        return val_proba, y_val
    oof_path = os.path.join(exp_dir, "oof", "oof_predictions.parquet")
    assert os.path.exists(oof_path), f"missing {oof_path} — run `python -m src.cv.run_cv --name {exp_name}` first"
    oof_proba = pd.read_parquet(oof_path)[labels].values
    y_oof = train[labels].values.astype(int)
    if source == "oof":
        return oof_proba, y_oof
    if source == "oof_val":
        return np.vstack([oof_proba, val_proba]), np.vstack([y_oof, y_val])
    raise ValueError(f"unknown threshold source: {source}")

def submit(cfg, exp_name, threshold_source="oof_val"):
    log = get_logger()
    labels, id_col, weights = cfg["columns"]["labels"], cfg["columns"]["id"], cfg["metric"]["weights"]
    train, val, test, sample, _ = build_data(cfg)
    exp_dir = os.path.join(cfg["paths"]["experiments_dir"], exp_name)
    test_p = pd.read_parquet(os.path.join(exp_dir, "test_proba.parquet"))
    grid = make_grid(cfg["threshold"]["grid_start"], cfg["threshold"]["grid_stop"], cfg["threshold"]["grid_num"])

    # tune thresholds on the chosen (robust) source
    tune_proba, tune_y = _threshold_data(cfg, exp_name, threshold_source, train, val, labels)
    thr = tune_thresholds(tune_y, tune_proba, grid)

    # always report the HELD-OUT val score as the honest compass
    val_proba = pd.read_parquet(os.path.join(exp_dir, "val_proba.parquet"))[labels].values
    y_val = val[labels].values.astype(int)
    val_score, per = competition_score(y_val, apply_thresholds(val_proba, thr), weights, labels)
    log.info(f"thresholds tuned on '{threshold_source}'  |  held-out val weighted-F1 = {val_score:.4f}")
    for j, l in enumerate(labels):
        log.info(f"  {l:24s} thr={thr[j]:.2f}  valF1={per[l]:.3f}  w={weights[l]}")

    pred = apply_thresholds(test_p[labels].values, thr)
    sub = pd.DataFrame({id_col: test_p[id_col].values})
    for j, l in enumerate(labels):
        sub[l] = pred[:, j]
    sub = sample[[id_col]].merge(sub, on=id_col, how="left")
    assert sub[labels].isna().sum().sum() == 0, "missing IDs vs sample submission"
    assert list(sub.columns) == [id_col] + labels
    assert set(np.unique(sub[labels].values)).issubset({0, 1})

    sdir = ensure_dir(cfg["paths"]["submissions_dir"])
    n = len(glob.glob(os.path.join(sdir, "submission_[0-9]*.csv"))) + 1
    out = os.path.join(sdir, f"submission_{n:03d}_{exp_name}.csv")
    sub.to_csv(out, index=False)
    save_json({"thresholds": dict(zip(labels, thr.tolist())), "threshold_source": threshold_source,
               "val_score": val_score}, os.path.join(exp_dir, "submission_thresholds.json"))

    log_path = os.path.join(sdir, "submission_log.csv")
    row = {"file": os.path.basename(out), "experiment": exp_name,
           "threshold_source": threshold_source, "val_score": round(val_score, 4),
           "thresholds": str([round(float(t), 2) for t in thr]), "zindi_public_score": ""}
    df = pd.concat([pd.read_csv(log_path), pd.DataFrame([row])], ignore_index=True) \
        if os.path.exists(log_path) else pd.DataFrame([row])
    df.to_csv(log_path, index=False)
    log.info(f"submission -> {out}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--thresholds", default="oof_val", choices=["val", "oof", "oof_val"])
    a = ap.parse_args()
    submit(load_config(a.config), a.name, threshold_source=a.thresholds)
