"""Schema + data-quality guards."""
from ..utils.logging import get_logger

def validate(train, val, test, sample, cfg, feats):
    log = get_logger()
    id_col, labels = cfg["columns"]["id"], cfg["columns"]["labels"]
    assert set(feats).issubset(test.columns), "feature mismatch between train and test"
    assert list(sample.columns) == [id_col] + labels, "submission columns mismatch"
    unseen = {c: sorted(set(test[c]) - set(train[c])) for c in feats}
    unseen = {k: v for k, v in unseen.items() if v}
    if unseen:
        log.info(f"unseen test categories (handled by OHE ignore): {unseen}")
    nan = int(train[feats].isna().sum().sum())
    log.info(f"schema OK | train NaNs={nan}")
    return unseen
