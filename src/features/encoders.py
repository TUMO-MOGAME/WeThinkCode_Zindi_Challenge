"""Preprocessing pipelines. Dense one-hot so every estimator (incl. HistGB) accepts it."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def build_preprocessor(features, method="onehot", handle_unknown="ignore"):
    if method == "onehot":
        return ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False), features)],
            remainder="drop",
        )
    raise ValueError(f"unknown encoding method: {method}")
