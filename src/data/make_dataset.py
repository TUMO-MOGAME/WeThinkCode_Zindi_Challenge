"""Load raw Zindi CSVs -> cleaned, typed interim parquet."""
import os
import pandas as pd
from ..utils.io import load_config, ensure_dir
from ..utils.logging import get_logger
from ..features.build_features import get_feature_cols

def load_raw(cfg):
    p = cfg["paths"]
    train = pd.read_csv(p["train_raw"])
    val   = pd.read_csv(p["val_raw"])
    test  = pd.read_csv(p["test_raw"])
    sample = pd.read_csv(p["sample_submission_raw"])
    return train, val, test, sample

def clean(df, features):
    df = df.copy()
    df[features] = df[features].astype(str)   # every value treated as a category
    return df

def build(cfg, write=True):
    train, val, test, sample = load_raw(cfg)
    feats = get_feature_cols(train, cfg["columns"]["id"], cfg["columns"]["labels"])
    train, val, test = clean(train, feats), clean(val, feats), clean(test, feats)
    if write:
        ip = ensure_dir(cfg["paths"]["interim_dir"])
        train.to_parquet(os.path.join(ip, "train.parquet"))
        val.to_parquet(os.path.join(ip, "val.parquet"))
        test.to_parquet(os.path.join(ip, "test.parquet"))
        sample.to_parquet(os.path.join(ip, "sample_submission.parquet"))
    return train, val, test, sample, feats

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    tr, va, te, ss, feats = build(cfg)
    get_logger().info(f"interim built | train {tr.shape} val {va.shape} test {te.shape} | {len(feats)} features")
