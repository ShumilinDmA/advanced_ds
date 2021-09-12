import numpy as np
import pandas as pd
from typing import Union
from sklearn.metrics import roc_auc_score, r2_score, precision_recall_curve, auc
from scipy import stats


# ===================== Classification metrics ===================== #

def gini_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro", sample_weight: np.ndarray = None,
               max_fpr: float = None, multi_class: str = "raise", labels: np.ndarray = None) -> float:
    """
    Get gini coefficient
    :param labels:  array-like of shape (n_classes,), default=None
                    Only used for multiclass targets. List of labels that index the
                    classes in ``y_score``. If ``None``, the numerical or lexicographical
                    order of the labels in ``y_true`` is used.
    :param multi_class: {'raise', 'ovr', 'ovo'}, default='raise'
                        Only used for multiclass targets. Determines the type of configuration
                        to use. The default value raises an error, so either
                        ``'ovr'`` or ``'ovo'`` must be passed explicitly.
    :param max_fpr: float > 0 and <= 1, default=None
                    If not ``None``, the standardized partial AUC [2]_ over the range
                    [0, max_fpr] is returned. For the multiclass case, ``max_fpr``,
                    should be either equal to ``None`` or ``1.0`` as AUC ROC partial
                    computation currently is not supported for multiclass.

    :param sample_weight: array-like of shape (n_samples,), default=None  Sample weights.
    :param average: {'micro', 'macro', 'samples', 'weighted'} or None, default='macro'
    :param y_true: True targets
    :param y_pred: Probability for class 1 in binary classification task
    :return: gini coefficient
    """
    return 2 * roc_auc_score(y_true, y_pred, average=average, sample_weight=sample_weight,
                             max_fpr=max_fpr, multi_class=multi_class, labels=labels) - 1


def pr_auc_score(y_true: np.ndarray, y_pred: np.ndarray,
                 pos_label: Union[int, str] = None, sample_weight: np.ndarray = None) -> float:
    """
    Return area under precision recall curve
    :param sample_weight: array-like of shape (n_samples,), default=None.
    :param pos_label: int or str, default=None
                      The label of the positive class. When pos_label=None, if y_true is in {-1, 1} or {0, 1},
                       pos_label is set to 1, otherwise an error will be raised.
    :param y_true: Targets for binary classification task
    :param y_pred: Probability for class 1 in binary classification task
    :return: area under precision recall curve
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight)
    precision = precision[:-1]  # to exclude point (0, 1)
    recall = recall[:-1]  # to exclude point (0, 1)
    pr_auc = auc(recall, precision)
    return pr_auc


def precision_at_recall_auc_score(y_true: np.ndarray, y_pred: np.ndarray, recall_threshold: float,
                                  pos_label: Union[int, str] = None, sample_weight: np.ndarray = None) -> float:
    """
    Area under precision recall curve with boundary condition under recall
    :param sample_weight: array-like of shape (n_samples,), default=None.
    :param pos_label: int or str, default=None
                      The label of the positive class. When pos_label=None, if y_true is in {-1, 1} or {0, 1},
                       pos_label is set to 1, otherwise an error will be raised.
    :param y_true: Targets for binary classification task
    :param y_pred: Probability for class 1 in binary classification task
    :param recall_threshold: Threshold for recall. All recall values less than Threshold will be included
    :return: precision at fixed recall area under curve
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight)
    precision = precision[:-1]  # to exclude point (0, 1)
    recall = recall[:-1]  # to exclude point (0, 1)

    mask = recall < recall_threshold
    recall = recall[mask]
    precision = precision[mask]

    pr_auc = auc(recall, precision)
    return pr_auc


# ===================== Regression metrics ===================== #

def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_features: float,
                      sample_weight: np.ndarray = None) -> float:
    """
    Get adjusted r2 score for regression task.
    Wiki: Coefficient of determination
    :param sample_weight: array-like of shape (n_samples,), default=None
    :param y_true: True targets
    :param y_pred: Model predictions
    :param n_features: Number of feature which is used in model
    :return: Adjusted r2 score
    """
    n_samples = len(y_true)
    r2 = r2_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    adjusted_r2 = 1 - (1 - r2) * ((n_samples - 1) / (n_samples - n_features - 1))
    return adjusted_r2


# ===================== Calibration metrics ===================== #

def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 10) -> float:
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


def adaptive_expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 10) -> float:
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

def mean_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Difference in mean values between y_true and y_pred
    :param y_true: Target
    :param y_pred: Predicted probability of class 1
    :return: error_rate
    """
    return y_true.mean() - y_pred.mean()


def psi_score(current_data: np.ndarray, expected_data: np.ndarray,
              bins: int, used_nan: Union[int, float, None] = None) -> float:
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


def vif_score(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
    """
    Variance inflation factor to estimate multicollinearity in features.
     If vif ~ 1 - there is no multicollinearity
     If vif ~ 5 - there is weakly multicollinearity
     If vif > 5+ - there is strong multicollinearity
    :param sample_weight: array-like of shape (n_samples,), default=None
    :param y_true: True data
    :param y_pred: Predicted value
    :return: VIF score for feature
    """
    vif = 1 / (1 - r2_score(y_true, y_pred, sample_weight=sample_weight) ** 2)
    return vif


def cramer_correlation(a_variable: Union[np.ndarray, pd.DataFrame],
                       b_variable: Union[np.ndarray, pd.DataFrame]) -> float:
    """
    Compute Cramer V coefficient between variable A and variable B
    :param a_variable: Categorical variable A
    :param b_variable: Categorical variable B
    :return:
    """
    _confusion_matrix = pd.crosstab(a_variable, b_variable)
    chi2 = stats.chi2_contingency(_confusion_matrix)[0]
    n = _confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = _confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((k_corr - 1), (r_corr - 1)))
