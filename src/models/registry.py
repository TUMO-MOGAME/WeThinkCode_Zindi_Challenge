"""The model zoo. get_estimator(name, params, seed) -> a base (single-output) classifier."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)

def _merge(default, params):
    out = dict(default); out.update(params or {}); return out

def get_estimator(name, params=None, seed=42):
    if name == "logreg":
        return LogisticRegression(**_merge(dict(max_iter=2000, class_weight="balanced"), params))
    if name == "rf":
        return RandomForestClassifier(**_merge(
            dict(n_estimators=400, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=seed), params))
    if name == "extratrees":
        return ExtraTreesClassifier(**_merge(
            dict(n_estimators=500, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=seed), params))
    if name == "histgb":
        return HistGradientBoostingClassifier(**_merge(
            dict(learning_rate=0.06, max_iter=400, l2_regularization=1.0, random_state=seed), params))
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(**_merge(
            dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=2.0, eval_metric="logloss", tree_method="hist", random_state=seed, n_jobs=-1), params))
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**_merge(
            dict(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=2.0, class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1), params))
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**_merge(
            dict(iterations=500, learning_rate=0.05, depth=5, l2_leaf_reg=3.0, random_seed=seed, verbose=0), params))
    raise ValueError(f"unknown model: {name}")

AVAILABLE = ["logreg", "rf", "extratrees", "histgb", "xgb", "lgbm", "catboost"]
