"""Assemble a full multi-label pipeline: preprocessor -> (OvR | ClassifierChain) head."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from .registry import get_estimator
from ..features.encoders import build_preprocessor

def make_model(name, features, strategy="ovr", params=None, seed=42,
               encoding="onehot", handle_unknown="ignore", n_labels=5, target_smoothing=20.0):
    pre = build_preprocessor(features, method=encoding, handle_unknown=handle_unknown,
                             n_labels=n_labels, smoothing=target_smoothing, seed=seed)
    est = get_estimator(name, params, seed)
    if strategy == "ovr":
        head = MultiOutputClassifier(est)
    elif strategy == "chain":
        head = ClassifierChain(est, order="random", random_state=seed)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    return Pipeline([("pre", pre), ("clf", head)])

def predict_proba_matrix(model, X):
    """Normalise predict_proba to an (n_samples, n_labels) matrix of P(gap=1)."""
    p = model.predict_proba(X)
    if isinstance(p, list):                      # MultiOutputClassifier
        return np.column_stack([pi[:, 1] for pi in p])
    return np.asarray(p)                          # ClassifierChain
