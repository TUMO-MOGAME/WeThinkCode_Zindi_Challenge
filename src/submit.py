"""Tune per-label thresholds on val.csv, apply to test, write a numbered, validated submission."""
import os, argparse, glob, time
import numpy as np, pandas as pd
from .utils.io import load_config, ensure_dir, save_json
from .utils.logging import get_logger
from .data.make_dataset import build as build_data
from .cv.evaluate import competition_score
from .postprocess.thresholds import tune_thresholds, apply_thresholds, make_grid

def submit(cfg, exp_name):
    log = get_logger()
    labels, id_col, weights = cfg["columns"]["labels"], cfg["columns"]["id"], cfg["metric"]["weights"]
    train, val, test, sample, _ = build_data(cfg)
    exp_dir = os.path.join(cfg["paths"]["experiments_dir"], exp_name)
    val_p  = pd.read_parquet(os.path.join(exp_dir, "val_proba.parquet"))
    test_p = pd.read_parquet(os.path.join(exp_dir, "test_proba.parquet"))
    grid = make_grid(cfg["threshold"]["grid_start"], cfg["threshold"]["grid_stop"], cfg["threshold"]["grid_num"])

    y_val = val[labels].values.astype(int)
    val_proba = val_p[labels].values
    thr = tune_thresholds(y_val, val_proba, grid)
    val_score, per = competition_score(y_val, apply_thresholds(val_proba, thr), weights, labels)
    log.info(f"val weighted-F1 (tuned thresholds) = {val_score:.4f}")
    for l, t in zip(labels, thr):
        log.info(f"  {l:24s} thr={t:.2f}  F1={per[l]:.3f}  w={weights[l]}")

    pred = apply_thresholds(test_p[labels].values, thr)
    sub = pd.DataFrame({id_col: test_p[id_col].values})
    for j, l in enumerate(labels):
        sub[l] = pred[:, j]
    sub = sample[[id_col]].merge(sub, on=id_col, how="left")
    assert sub[labels].isna().sum().sum() == 0, "missing IDs vs sample submission"
    assert list(sub.columns) == [id_col] + labels
    assert set(np.unique(sub[labels].values)).issubset({0, 1})

    sdir = ensure_dir(cfg["paths"]["submissions_dir"])
    n = len(glob.glob(os.path.join(sdir, "submission_*.csv"))) + 1
    out = os.path.join(sdir, f"submission_{n:03d}_{exp_name}.csv")
    sub.to_csv(out, index=False)
    save_json({"thresholds": dict(zip(labels, thr.tolist())), "val_score": val_score},
              os.path.join(exp_dir, "submission_thresholds.json"))

    # append to submission log
    log_path = os.path.join(sdir, "submission_log.csv")
    row = {"file": os.path.basename(out), "experiment": exp_name,
           "val_score": round(val_score, 4),
           "thresholds": str([round(float(t), 2) for t in thr]),
           "zindi_public_score": ""}   # fill in by hand after submitting
    df = pd.concat([pd.read_csv(log_path), pd.DataFrame([row])], ignore_index=True)         if os.path.exists(log_path) else pd.DataFrame([row])
    df.to_csv(log_path, index=False)
    log.info(f"submission -> {out}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    submit(load_config(a.config), a.name)
