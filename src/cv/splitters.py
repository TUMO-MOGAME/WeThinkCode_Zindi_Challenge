"""Cross-validation strategies. PRIMARY = multilabel-stratified (proper for 5 labels)."""
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

class _ComboStratified:
    """StratifiedKFold on the number of positive labels per row (a multi-label proxy)."""
    def __init__(self, n_splits, shuffle, seed):
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    def split(self, X, y):
        return self.skf.split(X, np.asarray(y).sum(axis=1))
    def get_n_splits(self, *a, **k):
        return self.skf.get_n_splits()

def get_splitter(name, n_splits, shuffle, seed):
    if name in ("multilabel_stratified", "mskf"):
        return MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    if name == "kfold":
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    if name in ("stratified_combo", "stratified_per_label"):
        return _ComboStratified(n_splits, shuffle, seed)
    raise ValueError(f"unknown cv strategy: {name}")
