"""Load a saved model, write val + test P(gap) probabilities into the experiment folder."""
import os, argparse
import joblib
import numpy as np, pandas as pd
from .utils.io import load_config, ensure_dir
from .utils.logging import get_logger
from .data.make_dataset import build as build_data
from .models.base import predict_proba_matrix

def predict(cfg, exp_name):
    log = get_logger()
    labels, id_col = cfg["columns"]["labels"], cfg["columns"]["id"]
    bundle = joblib.load(os.path.join(cfg["paths"]["models_dir"], f"{exp_name}.joblib"))
    model, feats = bundle["model"], bundle["features"]
    train, val, test, sample, _ = build_data(cfg)
    exp_dir = ensure_dir(os.path.join(cfg["paths"]["experiments_dir"], exp_name))
    for split_name, df in [("val", val), ("test", test)]:
        proba = predict_proba_matrix(model, df[feats])
        out = pd.DataFrame(proba, columns=labels); out.insert(0, id_col, df[id_col].values)
        out.to_parquet(os.path.join(exp_dir, f"{split_name}_proba.parquet"))
    log.info(f"wrote val_proba/test_proba -> {exp_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    predict(load_config(a.config), a.name)
