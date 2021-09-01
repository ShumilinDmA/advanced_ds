import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score, precision_recall_curve, auc, make_scorer


def gini_score(y_true: np.array, y_pred: np.array) -> float:
    """
    Get gini coefficient
    :param y_true: True targets
    :param y_pred: Probability for class 1 in binary classification task
    :return: gini coefficient
    """
    return 2*roc_auc_score(y_true, y_pred)-1


def adjusted_r2_score(y_true: np.array, y_pred: np.array, n_features: float) -> float:
    """
    Get adjusted r2 score for regression task.
    Wiki: Coefficient of determination
    :param y_true: True targets
    :param y_pred: Model predictions
    :param n_features: Number of feature which is used in model
    :return: Adjusted r2 score
    """
    n_samples = len(y_true)
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    adjusted_r2 = 1 - (1 - r2) * ((n_samples-1) / (n_samples - n_features - 1))
    return adjusted_r2


def pr_auc_score(y_true: np.array, y_pred: np.array) -> float:
    """
    Return area under precision recall curve
    :param y_true: Targets for binary classification task
    :param y_pred: Probability for class 1 in binary classification task
    :return: area under precision recall curve
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    precision = precision[:-1] # to exclude point (0, 1)
    recall = recall[:-1] # to exclude point (0, 1)
    pr_auc = auc(recall, precision)
    return pr_auc


def expected_calibration_error(y_true: np.array, y_pred: np.array, bins: int = 10) -> float:
    """
    Compute expected calibration error for binary classification task using histogram
    :param y_true: Targets
    :param y_pred: Probability of class 1
    :param bins: number of bins in histogram
    :return: ECE score
    """

    step =1/bins
    bins_list = np.arange(0, 1+step, step)
    bin_belongings = np.digitize(y_pred, bins=bins_list, right=True)

    per_bin_errors = []
    for bin_number in np.unique(bin_belongings):
        mask = bin_belongings == bin_number
        y_pred_bin = y_pred[mask].mean()
        y_true_bin = y_true[mask].mean()
        error = abs(y_true_bin - y_pred_bin)
        per_bin_errors.append(error)

    return np.mean(per_bin_errors)



def adaptive_expected_calibration_error(y_true: np.array, y_pred: np.array, bins: int = 10) -> float:
    """
    Compute adaptive expected calibration error for binary classification task using bins of equal sizes
    :param y_true: Targets
    :param y_pred: Probability of class 1
    :param bins: number of bins in histogram
    :return: adECE
    """

    def equal_bins(N, m):
        sep = (N.size / float(m)) * np.arange(1, m + 1)
        indx = sep.searchsorted(np.arange(N.size))
        return indx[N.argsort().argsort()]

    bin_belongings = equal_bins(y_pred, bins)

    per_bin_errors = []
    for bin_number in np.unique(bin_belongings):
        mask = bin_belongings == bin_number
        y_pred_bin = y_pred[mask].mean()
        y_true_bin = y_true[mask].mean()
        error = abs(y_true_bin - y_pred_bin)
        per_bin_errors.append(error)

    return np.mean(per_bin_errors)



def mean_calibration_error(y_true: np.array, y_pred: np.array) -> float:
    """
    Error which Alexander (Customer) wants to check as calibration error
    :param y_true: Target
    :param y_pred: Predicted probability of class 1
    :return: error_rate
    """
    return y_pred.mean() - y_true.mean()

