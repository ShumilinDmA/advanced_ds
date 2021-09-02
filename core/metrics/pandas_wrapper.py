import pandas as pd
from metric import *


# ===================== Classification metric wrappers ===================== #

def gini_score_pandas(subset: pd.DataFrame, y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> float:
    """
    Wrapper to get gini score for each group of data
    example: df.groupby(by="something").apply(gini_score_pandas, y_true_col="y_true", y_pred_col="y_pred")
    :param subset: Part of the dataset
    :param y_true_col: name of column which contain true labels
    :param y_pred_col: name of column which contain predictions for class 1
    :return: gini score
    """
    y_true = subset[y_true_col]
    y_pred = subset[y_pred_col]
    return gini_score(y_true, y_pred)


def pr_auc_score_pandas(subset: pd.DataFrame, y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> float:
    """
    Wrapper to get precision recall area under curve for each group of data
    example: df.groupby(by="something").apply(pr_auc_score_pandas, y_true_col="y_true", y_pred_col="y_pred")
    :param subset: Part of the dataset
    :param y_true_col: name of column which contain true labels
    :param y_pred_col: name of column which contain predictions for class 1
    :return: pr auc score
    """
    y_true = subset[y_true_col]
    y_pred = subset[y_pred_col]
    try:
        auc_val = pr_auc_score(y_true, y_pred)
    except ValueError:
        auc_val = 0
    return auc_val


def precision_at_recall_auc_score_pandas(subset: pd.DataFrame, recall_threshold: float,
                                         y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> float:
    """
    Wrapper to get precision recall area under curve for each group of data
    example: df.groupby(by="something").apply(precision_at_recall_auc_score_pandas, recall_threshold=0.05,
                                                y_true_col="y_true", y_pred_col="y_pred")
    :param subset: Part of the dataset
    :param recall_threshold: Threshold for recall
    :param y_true_col: name of column which contain true labels
    :param y_pred_col: name of column which contain predictions for class 1
    :return: area under shorted precision recall
    """
    y_true = subset[y_true_col]
    y_pred = subset[y_pred_col]
    return precision_at_recall_auc_score(y_true, y_pred, recall_threshold)


# ===================== Regression metric wrappers ===================== #

def adjusted_r2_score_pandas(subset: pd.DataFrame, n_features: int,
                             y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> float:
    """
    Wrapper to get adjusted r2 score for each group of data
    example: df.groupby(by="something").apply(gini_score_pandas, y_true_col="y_true", y_pred_col="y_pred")
    :param subset: Part of the dataset
    :param n_features: Number of features which is used for predictions
    :param y_true_col: name of column which contain true labels
    :param y_pred_col: name of column which contain predictions for class 1
    :return: adjusted r2 score
    """
    y_true = subset[y_true_col]
    y_pred = subset[y_pred_col]
    return adjusted_r2_score(y_true, y_pred, n_features=n_features)


# ===================== Calibration metric wrappers ===================== #

def expected_calibration_error_pandas(subset: pd.DataFrame, y_true_col: str = "y_true", y_pred_col: str = "y_pred",
                                      bins: int = 10) -> float:
    """
    Wrapper to get ece score for each group of data
    example: df.groupby(by="something").apply(expected_calibration_error_pandas,
                                                y_true_col="y_true", y_pred_col="y_pred", bins=10)
    :param subset: Part of the dataset
    :param y_true_col: name of column which contain true labels
    :param y_pred_col: name of column which contain predictions for class 1
    :param bins: Number of bins in histogram
    :return: ECE
    """
    y_true = subset[y_true_col]
    y_pred = subset[y_pred_col]
    return expected_calibration_error(y_true, y_pred, bins=bins)


def adaptive_expected_calibration_error_pandas(subset: pd.DataFrame, y_true_col: str = "y_true",
                                               y_pred_col: str = "y_pred", bins: int = 10) -> float:
    """
    Wrapper to adECE score for each group of data
    example: df.groupby(by="something").apply(adaptive_expected_calibration_error_pandas,
                                                y_true_col="y_true", y_pred_col="y_pred", bins=10)
    :param subset: Part of the dataset
    :param y_true_col: name of column which contain true labels
    :param y_pred_col: name of column which contain predictions for class 1
    :param bins: Number of bins in histogram
    :return: adECE
    """
    y_true = subset[y_true_col]
    y_pred = subset[y_pred_col]
    return adaptive_expected_calibration_error(y_true, y_pred, bins=bins)


# ===================== Auxiliary metric wrappers ===================== #

def mean_error_pandas(subset: pd.DataFrame, y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> float:
    """
    Wrapper to mean error score for each group of data
    example: df.groupby(by="something").apply(mean_error_pandas, y_true_col="y_true", y_pred_col="y_pred")
    :param subset: Part of the dataset
    :param y_true_col: name of column which contain true labels
    :param y_pred_col: name of column which contain predictions for class 1
    :return: error rate
    """
    y_true = subset[y_true_col]
    y_pred = subset[y_pred_col]
    return mean_error(y_true, y_pred)
