import pandas as pd
import numpy as np
from typing import Union, Any
from sklearn.metrics import make_scorer
from metric import *


gini_scorer = make_scorer(gini_score, greater_is_better=True, needs_proba=True, needs_threshold=False)

r2_scorer = make_scorer(r2_score, greater_is_better=True, needs_proba=False, needs_threshold=False)

pr_auc_scorer = make_scorer(pr_auc_score, greater_is_better=True, needs_proba=True, needs_threshold=False)

mean_error_scorer = make_scorer(mean_error, greater_is_better=False, needs_proba=False, needs_threshold=False)


def adjusted_r2_scorer(estimator: Any, x: Union[pd.DataFrame, np.array], y: Union[pd.Series, np.array]) -> float:
    y_pred = estimator.predict(x)
    adjusted_r2 = adjusted_r2_score(y, y_pred, n_features=x.shape[1])
    return adjusted_r2

