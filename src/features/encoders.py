"""Preprocessing pipelines. Dense one-hot so every estimator (incl. HistGB) accepts it.

Methods:
  onehot         - dense one-hot only (baseline)
  target         - leak-safe multi-label mean encoding only
  onehot_target  - both, concatenated
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from .target_encoding import MultiLabelTargetEncoder


def build_preprocessor(features, method="onehot", handle_unknown="ignore",
                       n_labels=5, smoothing=20.0, seed=42):
    parts = []
    if method in ("onehot", "onehot_target"):
        parts.append(("ohe", OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False), features))
    if method in ("target", "onehot_target"):
        parts.append(("te", MultiLabelTargetEncoder(features, n_labels=n_labels,
                                                     smoothing=smoothing, seed=seed), features))
    if not parts:
        raise ValueError(f"unknown encoding method: {method}")
    return ColumnTransformer(parts, remainder="drop")
