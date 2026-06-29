"""The EXACT Zindi metric. Used for every model-selection decision (not macro F1)."""
import numpy as np
from sklearn.metrics import f1_score

def per_label_f1(y_true, y_pred, labels):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {l: float(f1_score(y_true[:, j], y_pred[:, j], zero_division=0)) for j, l in enumerate(labels)}

def competition_score(y_true, y_pred, weights, labels):
    """Returns (overall_weighted_f1, {label: f1})."""
    f1s = per_label_f1(y_true, y_pred, labels)
    overall = float(sum(weights[l] * f1s[l] for l in labels))
    return overall, f1s
