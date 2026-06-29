"""Feature-column selection (and a home for future engineered features)."""

def get_feature_cols(df, id_col, labels):
    drop = {id_col, *labels}
    return [c for c in df.columns if c not in drop]
