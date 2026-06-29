"""Leak-safe multi-label target (mean) encoding.

For every (feature, label) pair we encode each category by the smoothed mean of that
label within the training data. Encoding is computed OUT-OF-FOLD on the training rows
(internal K-fold inside ``fit_transform``) so the model never sees a row's own label when
building its features; new data (``transform``) uses maps fit on the full training fold.
Because the outer CV fits this on the train fold only and applies it to the held-out fold,
there is no leakage across the outer fold boundary either.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


def _as_frame(X, features):
    if isinstance(X, pd.DataFrame):
        return X[features].astype(str).reset_index(drop=True)
    return pd.DataFrame(np.asarray(X), columns=features).astype(str)


class MultiLabelTargetEncoder(BaseEstimator, TransformerMixin):
    """Outputs n_features * n_labels smoothed mean-encoded columns."""

    def __init__(self, features, n_labels=5, smoothing=20.0, n_internal_splits=5, seed=42):
        self.features = features            # store verbatim (sklearn clone contract)
        self.n_labels = n_labels
        self.smoothing = smoothing
        self.n_internal_splits = n_internal_splits
        self.seed = seed

    def _build_maps(self, Xdf, Y):
        globals_ = Y.mean(axis=0)
        maps = {}
        for f in self.features:
            col = Xdf[f].values
            for li in range(self.n_labels):
                g = pd.DataFrame({"c": col, "y": Y[:, li]}).groupby("c")["y"]
                cnt, mean = g.count(), g.mean()
                enc = (cnt * mean + self.smoothing * globals_[li]) / (cnt + self.smoothing)
                maps[(f, li)] = enc
        return maps, globals_

    def _encode(self, Xdf, maps, globals_, rows=None):
        idx = range(len(Xdf)) if rows is None else rows
        sub = Xdf.iloc[idx]
        cols = []
        for f in self.features:
            s = sub[f]
            for li in range(self.n_labels):
                cols.append(s.map(maps[(f, li)]).fillna(globals_[li]).to_numpy())
        return np.column_stack(cols)

    @staticmethod
    def _as_Y(y):
        Y = np.asarray(y)
        return Y.reshape(-1, 1) if Y.ndim == 1 else Y

    def fit(self, X, y=None):
        Xdf = _as_frame(X, self.features)
        self.maps_, self.globals_ = self._build_maps(Xdf, self._as_Y(y))
        return self

    def transform(self, X):
        Xdf = _as_frame(X, self.features)
        return self._encode(Xdf, self.maps_, self.globals_)

    def fit_transform(self, X, y=None, **kw):
        Xdf = _as_frame(X, self.features)
        Y = self._as_Y(y)
        # full-data maps for later transform() on unseen rows
        self.maps_, self.globals_ = self._build_maps(Xdf, Y)
        # out-of-fold encoding for the training rows themselves
        out = np.empty((len(Xdf), len(self.features) * self.n_labels))
        kf = KFold(self.n_internal_splits, shuffle=True, random_state=self.seed)
        for tr, va in kf.split(Xdf):
            maps, globals_ = self._build_maps(Xdf.iloc[tr], Y[tr])
            out[va] = self._encode(Xdf, maps, globals_, rows=va)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"te_{f}_L{li}" for f in self.features for li in range(self.n_labels)])
