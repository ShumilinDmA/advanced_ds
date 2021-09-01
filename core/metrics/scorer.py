import pandas as pd
from sklearn.metrics import make_scorer, r2_score
from metric import gini_score, adjusted_r2_score


gini_scorer = make_scorer(gini_score, greater_is_better=True, needs_proba=True, needs_threshold=False)

r2_scorer = make_scorer(r2_score, greater_is_better=True, needs_proba=False, needs_threshold=False)


def adjusted_r2_scorer(estimator, X: pd.DataFrame, y: pd.Series) -> float:  # TODO FIx from pandas
    y_pred = estimator.predict(X)
    adjusted_r2 = adjusted_r2_score(y, y_pred, n_features=X.shape[1])
    return adjusted_r2

# TODO pr_auc_scorer