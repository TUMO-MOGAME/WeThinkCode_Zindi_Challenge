"""Fit the active model on full train.csv and persist model + metadata."""
import os, argparse
import joblib
from .utils.io import load_config, ensure_dir, save_json
from .utils.seed import seed_everything
from .utils.logging import get_logger
from .data.make_dataset import build as build_data
from .models.base import make_model

def train(cfg, mcfg, exp_name):
    log = get_logger(); seed = cfg["project"]["seed"]; seed_everything(seed)
    labels = cfg["columns"]["labels"]
    train, val, test, sample, feats = build_data(cfg)
    X, y = train[feats], train[labels].values.astype(int)
    name, strategy = mcfg["active"], mcfg.get("strategy", "ovr")
    params = mcfg.get(name, {}).get("params", {})
    enc = cfg.get("encoding", {})
    model = make_model(name, feats, strategy, params, seed,
                       encoding=enc.get("method", "onehot"),
                       handle_unknown=enc.get("handle_unknown", "ignore"),
                       n_labels=len(labels), target_smoothing=enc.get("target_smoothing", 20.0))
    model.fit(X, y)
    mdir = ensure_dir(cfg["paths"]["models_dir"])
    mpath = os.path.join(mdir, f"{exp_name}.joblib")
    joblib.dump({"model": model, "features": feats, "labels": labels,
                 "name": name, "strategy": strategy}, mpath)
    save_json({"experiment": exp_name, "model": name, "strategy": strategy,
               "params": params, "n_train": int(len(X)), "n_features": len(feats)},
              os.path.join(mdir, f"{exp_name}_metadata.json"))
    log.info(f"trained {name}/{strategy} on {len(X)} rows -> {mpath}")
    return mpath

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--model-config", default="config/model.yaml")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    cfg, mcfg = load_config(a.config), load_config(a.model_config)
    train(cfg, mcfg, a.name or f"exp_{mcfg['active']}_{mcfg.get('strategy','ovr')}")
