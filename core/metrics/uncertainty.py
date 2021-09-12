# TODO Poisson bootstrap for classical statistic and (TP,FP,TN,FN) with multiprocessing
import os
from typing import Callable, Union, Tuple
import numpy as np
import pandas as pd
import multiprocessing as mp


def generate_1d_metric(bundle: Tuple[Callable, Union[np.ndarray, pd.Series]]) -> float:
    """
    Compute metric with data subsample"
    :param bundle: Tuple with (metric_function, data)
    :return: Computed metric
    """
    metric_fn: Callable = bundle[0]
    data: Union[np.ndarray, pd.Series] = bundle[1]

    size = len(data)
    if isinstance(data, pd.Series):
        data = data.values()
    subsample_indx = np.random.choice(np.arange(data), size=size, replace=True)
    metric = metric_fn(data[subsample_indx])
    return metric


def bootstrap_confidence_intervals_1d(data: Union[np.ndarray, pd.Series],
                                      metric_fn: Callable,
                                      b: int,
                                      alpha: float,
                                      n_jobs: Union[int, None]) -> Tuple[float, float, float]:
    """
    Compute classical bootstrap confidence intervals
    :param data: (n_samples, 1) array with data
    :param metric_fn: callable function to compute metric
    :param b: number of bootstraps rounds
    :param alpha: confidence level
    :param n_jobs: number of process
    :return: Tuple[left_bound, mean_value, right_bound]
    """

    if n_jobs == -1:
        n_jobs_selected = max(os.cpu_count() - 1, 1)
    elif n_jobs is None:
        n_jobs_selected = 1
    else:
        n_jobs_selected = n_jobs

    data_generator = ((metric_fn, data) for _ in range(b))

    mp.set_start_method("fork", force=True)
    with mp.Pool(processes=n_jobs_selected) as pool:
        metrics_list = np.array(list(pool.imap(generate_1d_metric, data_generator)))

    metrics_list.sort()
    left_bound = np.percentile(metrics_list, 100 * (1 - alpha) / 2)
    right_bound = np.percentile(metrics_list, 100 * (alpha + (1-alpha) / 2))
    mean_value = np.mean(metrics_list)

    return left_bound, mean_value, right_bound


def generate_2d_metric(bundle: Tuple[Callable,  Union[np.ndarray, pd.Series],  Union[np.ndarray, pd.Series]]) -> float:
    """
    Compute metric with data subsample according each other"
    :param bundle: Tuple with (metric_function, data1, data2) or (metric_function, y_true, y_pred)
    :return: Computed metric
    """
    metric_fn: Callable = bundle[0]
    y_true: Union[np.ndarray, pd.Series] = bundle[1]
    y_pred: Union[np.ndarray, pd.Series] = bundle[2]

    size = len(y_true)
    if isinstance(y_true, pd.Series):
        y_true = y_true.values()
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    subsample_indx = np.random.choice(np.arange(size), size=size, replace=True)
    metric = metric_fn(y_true[subsample_indx], y_pred[subsample_indx])
    return metric


def bootstrap_confidence_intervals_2d(y_true: Union[np.ndarray, pd.Series],
                                      y_pred: Union[np.ndarray, pd.Series],
                                      metric_fn: Callable,
                                      b: int,
                                      alpha: float,
                                      n_jobs: Union[int, None]) -> Tuple[float, float, float]:
    """
    Compute classical bootstrap confidence intervals
    :param y_true: (n_samples, 1) array with data, according with y_pred
    :param y_pred: (n_samples, 1) array with data, according with y_true
    :param metric_fn: callable function to compute metric
    :param b: number of bootstraps rounds
    :param alpha: confidence level
    :param n_jobs: number of process
    :return: Tuple[left_bound, mean_value, right_bound]
    """

    if n_jobs == -1:
        n_jobs_selected = max(os.cpu_count() - 1, 1)
    elif n_jobs is None:
        n_jobs_selected = 1
    else:
        n_jobs_selected = n_jobs

    data_generator = ((metric_fn, y_true, y_pred) for _ in range(b))

    mp.set_start_method("fork", force=True)
    with mp.Pool(processes=n_jobs_selected) as pool:
        metrics_list = np.array(list(pool.imap(generate_2d_metric, data_generator)))

    metrics_list.sort()
    left_bound = np.percentile(metrics_list, 100 * (1 - alpha) / 2)
    right_bound = np.percentile(metrics_list, 100 * (alpha + (1-alpha) / 2))
    mean_value = np.mean(metrics_list)

    return left_bound, mean_value, right_bound
