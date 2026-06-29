"""Optuna search for the active model, optimising the exact CV weighted-F1."""
import argparse
import numpy as np
import optuna
from .utils.io import load_config, save_json
from .utils.seed import seed_everything
from .utils.logging import get_logger
from .data.make_dataset import build as build_data
from .models.base import make_model, predict_proba_matrix
from .cv.evaluate import competition_score
from .cv.splitters import get_splitter
from .postprocess.thresholds import tune_thresholds, apply_thresholds, make_grid

def suggest_params(trial, search):
    out = {}
    for k, spec in (search or {}).items():
        t = spec["type"]
        if t == "float":
            out[k] = trial.suggest_float(k, spec["low"], spec["high"], log=spec.get("log", False))
        elif t == "int":
            out[k] = trial.suggest_int(k, spec["low"], spec["high"])
        elif t == "cat":
            out[k] = trial.suggest_categorical(k, spec["choices"])
    return out

def tune(cfg, mcfg, trials=30, cv_splits=3):
    log = get_logger(); seed = cfg["project"]["seed"]; seed_everything(seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    labels, weights = cfg["columns"]["labels"], cfg["metric"]["weights"]
    train, val, test, sample, feats = build_data(cfg)
    X, y = train[feats], train[labels].values.astype(int)
    name, strategy = mcfg["active"], mcfg.get("strategy", "ovr")
    search = mcfg.get(name, {}).get("search", {})
    grid = make_grid(cfg["threshold"]["grid_start"], cfg["threshold"]["grid_stop"], cfg["threshold"]["grid_num"])

    def objective(trial):
        params = suggest_params(trial, search)
        splitter = get_splitter(cfg["cv"]["primary"], cv_splits, cfg["cv"]["shuffle"], seed)
        oof = np.zeros((len(X), len(labels)))
        for tr, va in splitter.split(X, y):
            m = make_model(name, feats, strategy, params, seed)
            m.fit(X.iloc[tr], y[tr])
            oof[va] = predict_proba_matrix(m, X.iloc[va])
        thr = tune_thresholds(y, oof, grid)
        return competition_score(y, apply_thresholds(oof, thr), weights, labels)[0]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, show_progress_bar=True)
    log.info(f"best CV weighted-F1 = {study.best_value:.4f}")
    log.info(f"best params = {study.best_params}")
    save_json({"model": name, "strategy": strategy, "best_value": study.best_value,
               "best_params": study.best_params},
              f"experiments/tune_{name}_{strategy}.json")
    return study.best_params

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--model-config", default="config/model.yaml")
    ap.add_argument("--trials", type=int, default=30)
    a = ap.parse_args()
    tune(load_config(a.config), load_config(a.model_config), trials=a.trials)
