"""OOF cross-validation driver. Writes a full experiment folder under experiments/<name>/."""
import os, argparse
import numpy as np, pandas as pd
from ..utils.io import load_config, ensure_dir, save_json
from ..utils.seed import seed_everything
from ..utils.logging import get_logger
from ..data.make_dataset import build as build_data
from ..data.validate import validate
from ..models.base import make_model, predict_proba_matrix
from .evaluate import competition_score
from .splitters import get_splitter
from ..postprocess.thresholds import tune_thresholds, apply_thresholds, make_grid

def _oof(cfg, mcfg, feats, X, y, splitter_name, grid, weights, labels, seed, log):
    splitter = get_splitter(splitter_name, cfg["cv"]["n_splits"], cfg["cv"]["shuffle"], seed)
    name, strategy = mcfg["active"], mcfg.get("strategy", "ovr")
    params = mcfg.get(name, {}).get("params", {})
    enc = cfg.get("encoding", {})
    oof = np.zeros((len(X), len(labels))); rows = []
    for k, (tr, va) in enumerate(splitter.split(X, y)):
        m = make_model(name, feats, strategy, params, seed,
                       encoding=enc.get("method", "onehot"),
                       handle_unknown=enc.get("handle_unknown", "ignore"),
                       n_labels=len(labels), target_smoothing=enc.get("target_smoothing", 20.0))
        m.fit(X.iloc[tr], y[tr])
        oof[va] = predict_proba_matrix(m, X.iloc[va])
        thr = tune_thresholds(y[va], oof[va], grid)
        ptr = predict_proba_matrix(m, X.iloc[tr])
        tr_thr = tune_thresholds(y[tr], ptr, grid)
        s_va, _ = competition_score(y[va], apply_thresholds(oof[va], thr), weights, labels)
        s_tr, _ = competition_score(y[tr], apply_thresholds(ptr, tr_thr), weights, labels)
        rows.append({"fold": k, "val_score": s_va, "train_score": s_tr, "gap": s_tr - s_va})
        log.info(f"[{splitter_name}] fold {k}: val={s_va:.4f} train={s_tr:.4f} gap={s_tr-s_va:.3f}")
    thr = tune_thresholds(y, oof, grid)
    overall, per = competition_score(y, apply_thresholds(oof, thr), weights, labels)
    return oof, thr, overall, per, pd.DataFrame(rows)

def run_cv(cfg, mcfg, exp_name, secondary=True):
    log = get_logger(); seed = cfg["project"]["seed"]; seed_everything(seed)
    labels, id_col, weights = cfg["columns"]["labels"], cfg["columns"]["id"], cfg["metric"]["weights"]
    train, val, test, sample, feats = build_data(cfg)
    validate(train, val, test, sample, cfg, feats)
    X, y = train[feats], train[labels].values.astype(int)
    grid = make_grid(cfg["threshold"]["grid_start"], cfg["threshold"]["grid_stop"], cfg["threshold"]["grid_num"])
    exp_dir = ensure_dir(os.path.join(cfg["paths"]["experiments_dir"], exp_name))

    oof, thr, overall, per, fold_df = _oof(cfg, mcfg, feats, X, y, cfg["cv"]["primary"], grid, weights, labels, seed, log)
    fold_df.to_csv(os.path.join(exp_dir, "fold_scores.csv"), index=False)
    ensure_dir(os.path.join(exp_dir, "oof"))
    pd.DataFrame(oof, columns=labels).assign(**{id_col: train[id_col].values})         .to_parquet(os.path.join(exp_dir, "oof", "oof_predictions.parquet"))
    pd.Series(per, name="f1").to_csv(os.path.join(exp_dir, "per_label_score.csv"))
    save_json(dict(zip(labels, thr.tolist())), os.path.join(exp_dir, "thresholds.json"))
    save_json({"cv_strategy": cfg["cv"]["primary"], "overall_weighted_f1": overall,
               "per_label_f1": per, "thresholds": dict(zip(labels, thr.tolist())),
               "model": mcfg["active"], "strategy": mcfg.get("strategy", "ovr"),
               "train_cv_gap": float(fold_df["gap"].mean())},
              os.path.join(exp_dir, "overall_score.json"))

    if secondary:
        for sname in cfg["cv"].get("secondary", []):
            _, _, so, sp, sdf = _oof(cfg, mcfg, feats, X, y, sname, grid, weights, labels, seed, log)
            sd = ensure_dir(os.path.join(exp_dir, "secondary", sname))
            sdf.to_csv(os.path.join(sd, "fold_scores.csv"), index=False)
            save_json({"overall_weighted_f1": so, "per_label_f1": sp}, os.path.join(sd, "overall_score.json"))

    log.info(f"PRIMARY CV weighted-F1 = {overall:.4f}  (gap {fold_df['gap'].mean():.3f}) -> {exp_dir}")
    return overall

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--model-config", default="config/model.yaml")
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-secondary", action="store_true")
    a = ap.parse_args()
    cfg, mcfg = load_config(a.config), load_config(a.model_config)
    exp = a.name or f"exp_{mcfg['active']}_{mcfg.get('strategy','ovr')}"
    run_cv(cfg, mcfg, exp, secondary=not a.no_secondary)
