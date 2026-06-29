"""Probability-level blending of several fitted multi-label pipelines."""
import numpy as np
from .base import predict_proba_matrix

def blend_proba(models, X, weights=None):
    probas = [predict_proba_matrix(m, X) for m in models]
    if weights is None:
        return np.mean(probas, axis=0)
    w = np.asarray(weights, dtype=float); w = w / w.sum()
    return np.tensordot(w, np.stack(probas), axes=(0, 0))
