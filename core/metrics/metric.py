import numpy as np
from typing import Optional
from sklearn.metrics import roc_auc_score, r2_score, precision_recall_curve, auc


# ===================== Classification metrics ===================== #

def gini_score(y_true: np.array, y_pred: np.array) -> float:
    """
    Get gini coefficient
    :param y_true: True targets
    :param y_pred: Probability for class 1 in binary classification task
    :return: gini coefficient
    """
    return 2 * roc_auc_score(y_true, y_pred) - 1


def pr_auc_score(y_true: np.array, y_pred: np.array) -> float:
    """
    Return area under precision recall curve
    :param y_true: Targets for binary classification task
    :param y_pred: Probability for class 1 in binary classification task
    :return: area under precision recall curve
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    precision = precision[:-1]  # to exclude point (0, 1)
    recall = recall[:-1]  # to exclude point (0, 1)
    pr_auc = auc(recall, precision)
    return pr_auc


def precision_at_recall_auc_score(y_true: np.array, y_pred: np.array, recall_threshold) -> float:
    """
    Area under precision recall curve with boundary condition under recall
    :param y_true: Targets for binary classification task
    :param y_pred: Probability for class 1 in binary classification task
    :param recall_threshold: Threshold for recall. All recall values less than Threshold will be included
    :return: precision at fixed recall area under curve
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    precision = precision[:-1]  # to exclude point (0, 1)
    recall = recall[:-1]  # to exclude point (0, 1)

    mask = recall < recall_threshold
    recall = recall[mask]
    precision = precision[mask]

    pr_auc = auc(recall, precision)
    return pr_auc


# ===================== Regression metrics ===================== #

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
    adjusted_r2 = 1 - (1 - r2) * ((n_samples - 1) / (n_samples - n_features - 1))
    return adjusted_r2


# ===================== Calibration metrics ===================== #

def expected_calibration_error(y_true: np.array, y_pred: np.array, bins: int = 10) -> float:
    """
    Compute expected calibration error for binary classification task using histogram
    :param y_true: Targets
    :param y_pred: Probability of class 1
    :param bins: number of bins in histogram
    :return: ECE score
    """

    step = 1 / bins
    bins_list = np.arange(0, 1 + step, step)
    bin_belongings = np.digitize(y_pred, bins=bins_list, right=True)

    per_bin_errors = []
    for bin_number in np.unique(bin_belongings):
        mask = bin_belongings == bin_number
        y_pred_bin = y_pred[mask].mean()
        y_true_bin = y_true[mask].mean()
        error = abs(y_true_bin - y_pred_bin) * np.sum(mask)
        per_bin_errors.append(error)

    return np.sum(per_bin_errors) / len(y_true)


def adaptive_expected_calibration_error(y_true: np.array, y_pred: np.array, bins: int = 10) -> float:
    """
    Compute adaptive expected calibration error for binary classification task using bins of equal sizes
    :param y_true: Targets
    :param y_pred: Probability of class 1
    :param bins: number of bins in histogram
    :return: adECE
    """

    def equal_bins(n, m):
        sep = (n.size / float(m)) * np.arange(1, m + 1)
        indx = sep.searchsorted(np.arange(n.size))
        return indx[n.argsort().argsort()]

    bin_belongings = equal_bins(y_pred, bins)

    per_bin_errors = []
    for bin_number in np.unique(bin_belongings):
        mask = bin_belongings == bin_number
        y_pred_bin = y_pred[mask].mean()
        y_true_bin = y_true[mask].mean()
        error = abs(y_true_bin - y_pred_bin) * np.sum(mask)
        per_bin_errors.append(error)

    return np.sum(per_bin_errors) / len(y_true)


# ===================== Auxiliary metrics ===================== #

def mean_error(y_true: np.array, y_pred: np.array) -> float:
    """
    Difference in mean values between y_true and y_pred
    :param y_true: Target
    :param y_pred: Predicted probability of class 1
    :return: error_rate
    """
    return y_true.mean() - y_pred.mean()


def psi_score(current_data: np.array, expected_data: np.array, bins: int, used_nan: Optional = None) -> float:
    """
    Compute population stability index over current data and expected data
    :param current_data: Data in current experiment
    :param expected_data: Data in training algorithm
    :param bins: Number of bins in histogram
    :param used_nan: If nan exist - place value here, if not set None, Default - None
    :return:
    """

    def fill_zeros(array):
        array_back = np.where(array == 0, 1e-8, array)
        return array_back

    def scale_range(inputs, min_val, max_val):
        """
        Rescale bin's edges
        :param inputs: bins edges
        :param min_val: max value in data
        :param max_val: min value in data
        :return: Scaled bin's edges
        """
        inputs += np.min(inputs)
        inputs /= np.max(inputs) / (max_val - min_val)
        inputs += min_val
        return inputs

    nan_mask_current = []
    nan_mask_expected = []
    if used_nan is not None:
        nan_mask_current = (current_data == used_nan)
        nan_mask_expected = (expected_data == used_nan)

    current_data = current_data[~nan_mask_current]
    expected_data = expected_data[~nan_mask_expected]

    breakpoints = np.arange(0, bins + 1) / (bins * 100)
    breakpoints = scale_range(breakpoints, np.min(expected_data), np.max(expected_data))

    expected_bins, expected_bins_edges = np.histogram(expected_data, bins=breakpoints)
    expected_bins = expected_bins / len(expected_data)
    expected_bins = np.hstack((expected_bins, [np.sum(nan_mask_expected) / len(nan_mask_expected)]))  # add nans

    current_bins, _ = np.histogram(current_data, bins=breakpoints)
    current_bins = current_bins / len(current_data)
    current_bins = np.hstack((current_bins, np.sum(nan_mask_current) / len(nan_mask_current)))  # add nans

    current_bins = fill_zeros(current_bins)
    expected_bins = fill_zeros(expected_bins)

    psi_bins = (current_bins - expected_bins) * np.log(current_bins / expected_bins)
    psi = psi_bins.sum()

    return psi


def vif_score(y_true: np.array, y_pred: np.array) -> float:
    """
    Variance inflation factor to estimate multicollinearity in features.
     If vif ~ 1 - there is no multicollinearity
     If vif ~ 5 - there is weakly multicollinearity
     If vif > 5+ - there is strong multicollinearity
    :param y_true: True data
    :param y_pred: Predicted value
    :return: VIF score for feature
    """
    vif = 1 / (1 - r2_score(y_true, y_pred) ** 2)
    return vif


# TODO maybe information value (IV), maybe Cramer V coefficient
