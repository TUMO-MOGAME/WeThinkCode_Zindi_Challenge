"""Per-label F1-maximising threshold tuning (the highest-leverage step)."""
import numpy as np
from sklearn.metrics import f1_score

def make_grid(start, stop, num):
    return np.round(np.linspace(start, stop, num), 4)

def tune_thresholds(y_true, proba, grid):
    """Independent per-label threshold; optimal because the weighted score is separable per label."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    thr = np.empty(proba.shape[1])
    for j in range(proba.shape[1]):
        f1s = [f1_score(y_true[:, j], (proba[:, j] >= t).astype(int), zero_division=0) for t in grid]
        thr[j] = grid[int(np.argmax(f1s))]
    return thr

def apply_thresholds(proba, thr):
    return (np.asarray(proba) >= np.asarray(thr)).astype(int)
