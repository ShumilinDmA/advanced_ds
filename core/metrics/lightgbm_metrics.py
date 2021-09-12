from typing import Tuple, Union
import numpy as np
import lightgbm as lgbm
from metrics import adjusted_r2_score, gini_score, pr_auc_score
from sklearn.metrics import f1_score, fbeta_score


def sigmoid(data: np.ndarray):
    """Compute sigmoid values for each sample of data"""
    return 1 / (1 + np.exp(-data) + 1e-7)


def softmax(data: np.ndarray):
    """Compute softmax values for each sets of scores in data."""
    e_x = np.exp(data - np.max(data))
    return e_x / e_x.sum(axis=0)


def check_shape_and_transform_to_scores(data: np.ndarray) -> np.ndarray:
    """
    Apply sigmoid or softmax transformation
    :param data: np.ndarray with logits
    :return: Probability scores
    """
    if data.shape[1] == 1:
        return sigmoid(data)
    if data.shape[1] > 1:
        return softmax(data)


def check_shape_and_transform_to_labels(data: np.ndarray, threshold: Union[float, None] = 0.5) -> np.ndarray:
    """
    Apply sigmoid or softmax transformation and use threshold to convert it to labels
    :param data: np.ndarray with logits
    :param threshold: Optional, None for softmax, float for sigmoid, default 0.5
    :return: Labels
    """
    if data.shape[1] == 1:
        return np.array(sigmoid(data) > threshold).astype(int)
    if data.shape[1] > 1:
        return np.argmax(softmax(data), axis=-1)


class AdjustedR2Lightgbm:
    """
    Metric to use in LightGBM during training
    Example:
        ```
        lgbm_metric = AdjustedR2LightGBM(n_features=len(feature_list))

        lgbm.train(params=params, train_set=dtrain, feval=lgbm_metric)
        OR
        params = {"metric": lgbm_metric}
        model = lgbm.LGBMRegressor(**params)
        ```
    """

    def __init__(self, n_features: int, sample_weight: np.ndarray = None) -> None:
        self.n_features = n_features
        self.sample_weight = sample_weight

    def __call__(self, arg1: np.ndarray, arg2: Union[np.ndarray, lgbm.basic.Dataset]) -> Tuple[str, float, bool]:

        if isinstance(arg1, np.ndarray) and isinstance(arg2, np.ndarray):
            return self._sklearn_api(y_true=arg1, y_pred=arg2)
        if isinstance(arg1, np.ndarray) and isinstance(arg2, lgbm.basic.Dataset):
            return self._independent_api(y_pred=arg1, data=arg2)
        assert TypeError, "Type of inputs should be np.ndarray or lightgbm.basic.Dataset."

    def _sklearn_api(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        score = adjusted_r2_score(y_true=y_true, y_pred=y_pred, n_features=self.n_features,
                                  sample_weight=self.sample_weight)
        return "adj_r2", score, True

    def _independent_api(self, y_pred: np.ndarray, data: lgbm.basic.Dataset) -> Tuple[str, float, bool]:
        y_true = data.get_label()
        score = adjusted_r2_score(y_true=y_true, y_pred=y_pred, n_features=self.n_features,
                                  sample_weight=self.sample_weight)
        return "adj_r2", score, True


class GiniLightgbm:
    """
    Metric to use in LightGBM during training
    Example:
        ```
        lgbm_metric = GiniLightgbm()
        lgbm.train(params=params, train_set=dtrain, feval=lgbm_metric)
        OR
        params = {"metric": lgbm_metric}
        model = lgbm.LGBMClassifier(**params)
        ```
    """
    def __init__(self, average: str = "macro", sample_weight: np.ndarray = None,
                 max_fpr: float = None, multi_class: str = "raise", labels: np.ndarray = None):
        self.average = average
        self.sample_weight = sample_weight
        self.max_fpr = max_fpr
        self.multi_class = multi_class
        self.labels = labels

    def __call__(self, arg1: np.ndarray, arg2: Union[np.ndarray, lgbm.basic.Dataset]) -> Tuple[str, float, bool]:

        if isinstance(arg1, np.ndarray) and isinstance(arg2, np.ndarray):
            return self._sklearn_api(y_true=arg1, y_pred=arg2)
        if isinstance(arg1, np.ndarray) and isinstance(arg2, lgbm.basic.Dataset):
            return self._independent_api(y_pred=arg1, data=arg2)
        assert TypeError, "Type of inputs should be np.ndarray or lightgbm.basic.Dataset."

    def _sklearn_api(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        y_pred = check_shape_and_transform_to_scores(y_pred)

        score = gini_score(y_true=y_true, y_pred=y_pred, average=self.average, sample_weight=self.sample_weight,
                           max_fpr=self.max_fpr, multi_class=self.multi_class, labels=self.labels)
        return "gini", score, True

    def _independent_api(self, y_pred: np.ndarray, data: lgbm.basic.Dataset) -> Tuple[str, float, bool]:
        y_true = data.get_label()
        y_pred = check_shape_and_transform_to_scores(y_pred)

        score = gini_score(y_true=y_true, y_pred=y_pred, average=self.average, sample_weight=self.sample_weight,
                           max_fpr=self.max_fpr, multi_class=self.multi_class, labels=self.labels)
        return "gini", score, True


class AUPRCLightgbm:
    """
    Metric to use in LightGBM during training
    Example:
        ```
        lgbm_metric = AUPRCLightgbm()
        lgbm.train(params=params, train_set=dtrain, feval=lgbm_metric)
        OR
        params = {"metric": lgbm_metric}
        model = lgbm.LGBMClassifier(**params)
        ```
    """

    def __init__(self, pos_label: Union[int, str] = None, sample_weight: np.ndarray = None) -> None:
        self.pos_label = pos_label
        self.sample_weight = sample_weight

    def __call__(self, arg1: np.ndarray, arg2: Union[np.ndarray, lgbm.basic.Dataset]) -> Tuple[str, float, bool]:

        if isinstance(arg1, np.ndarray) and isinstance(arg2, np.ndarray):
            return self._sklearn_api(y_true=arg1, y_pred=arg2)
        if isinstance(arg1, np.ndarray) and isinstance(arg2, lgbm.basic.Dataset):
            return self._independent_api(y_pred=arg1, data=arg2)
        assert TypeError, "Type of inputs should be np.ndarray or lightgbm.basic.Dataset."

    def _sklearn_api(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        y_pred = check_shape_and_transform_to_scores(y_pred)

        score = pr_auc_score(y_true=y_true, y_pred=y_pred, sample_weight=self.sample_weight, pos_label=self.pos_label)
        return "pr_auc", score, True

    def _independent_api(self, y_pred: np.ndarray, data: lgbm.basic.Dataset) -> Tuple[str, float, bool]:
        y_true = data.get_label()
        y_pred = check_shape_and_transform_to_scores(y_pred)

        score = pr_auc_score(y_true=y_true, y_pred=y_pred, sample_weight=self.sample_weight, pos_label=self.pos_label)
        return "pr_auc", score, True


class F1Lightgbm:
    """
    Metric to use in LightGBM during training
    Example:
        ```
        lgbm_metric = F1Lightgbm()
        lgbm.train(params=params, train_set=dtrain, feval=lgbm_metric)
        OR
        params = {"metric": lgbm_metric}
        model = lgbm.LGBMClassifier(**params)
        ```
    """
    def __init__(self, threshold: float, labels=None, pos_label=1, average='binary',
                 sample_weight=None, zero_division="warn") -> None:

        self.threshold = threshold
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.zero_division = zero_division

    def __call__(self, arg1: np.ndarray, arg2: Union[np.ndarray, lgbm.basic.Dataset]) -> Tuple[str, float, bool]:

        if isinstance(arg1, np.ndarray) and isinstance(arg2, np.ndarray):
            return self._sklearn_api(y_true=arg1, y_pred=arg2)
        if isinstance(arg1, np.ndarray) and isinstance(arg2, lgbm.basic.Dataset):
            return self._independent_api(y_pred=arg1, data=arg2)
        assert TypeError, "Type of inputs should be np.ndarray or lightgbm.basic.Dataset."

    def _sklearn_api(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        y_pred = check_shape_and_transform_to_labels(y_pred, self.threshold)

        score = f1_score(y_true=y_true, y_pred=y_pred, labels=self.labels, average=self.average,
                         sample_weight=self.sample_weight, zero_division=self.zero_division, pos_label=self.pos_label)
        return "f1", score, True

    def _independent_api(self, y_pred: np.ndarray, data: lgbm.basic.Dataset) -> Tuple[str, float, bool]:
        y_true = data.get_label()
        y_pred = check_shape_and_transform_to_labels(y_pred, self.threshold)

        score = f1_score(y_true=y_true, y_pred=y_pred, labels=self.labels, average=self.average,
                         sample_weight=self.sample_weight, zero_division=self.zero_division, pos_label=self.pos_label)
        return "f1", score, True


class FbetaLightgbm:
    """
    Metric to use in LightGBM during training
    Example:
        ```
        lgbm_metric = FbetaLightgbm()
        lgbm.train(params=params, train_set=dtrain, feval=lgbm_metric)
        OR
        params = {"metric": lgbm_metric}
        model = lgbm.LGBMClassifier(**params)
        ```
    """
    def __init__(self, threshold: float, beta: float, labels=None, pos_label=1,
                 average='binary', sample_weight=None, zero_division="warn") -> None:
        self.threshold = threshold
        self.beta = beta
        self.labels = labels
        self.average = average
        self.sample_weight = sample_weight
        self.zero_division = zero_division
        self.pos_label = pos_label

    def __call__(self, arg1: np.ndarray, arg2: Union[np.ndarray, lgbm.basic.Dataset]) -> Tuple[str, float, bool]:

        if isinstance(arg1, np.ndarray) and isinstance(arg2, np.ndarray):
            return self._sklearn_api(y_true=arg1, y_pred=arg2)
        if isinstance(arg1, np.ndarray) and isinstance(arg2, lgbm.basic.Dataset):
            return self._independent_api(y_pred=arg1, data=arg2)
        assert TypeError, "Type of inputs should be np.ndarray or lightgbm.basic.Dataset."

    def _sklearn_api(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        y_pred = check_shape_and_transform_to_labels(y_pred, self.threshold)

        score = fbeta_score(y_true=y_true, y_pred=y_pred, beta=self.beta, labels=self.labels, average=self.average,
                            sample_weight=self.sample_weight, zero_division=self.zero_division,
                            pos_label=self.pos_label)
        return "fbeta", score, True

    def _independent_api(self, y_pred: np.ndarray, data: lgbm.basic.Dataset) -> Tuple[str, float, bool]:
        y_true = data.get_label()
        y_pred = check_shape_and_transform_to_labels(y_pred, self.threshold)

        score = fbeta_score(y_true=y_true, y_pred=y_pred, beta=self.beta, labels=self.labels, average=self.average,
                            sample_weight=self.sample_weight, zero_division=self.zero_division,
                            pos_label=self.pos_label)
        return "fbeta", score, True


class BinaryFocalLossErrorLightgbm:
    def __init__(self, alpha: float, gamma: float) -> None:
        self.a = alpha
        self.g = gamma

    def __call__(self, arg1: np.ndarray, arg2: Union[np.ndarray, lgbm.basic.Dataset]) -> Tuple[str, float, bool]:
        if isinstance(arg1, np.ndarray) and isinstance(arg2, np.ndarray):
            return self._sklearn_api(y_true=arg1, y_pred=arg2)
        if isinstance(arg1, np.ndarray) and isinstance(arg2, lgbm.basic.Dataset):
            return self._independent_api(y_pred=arg1, data=arg2)
        assert TypeError, "Type of inputs should be np.ndarray or lightgbm.basic.Dataset."

    def _sklearn_api(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        p = check_shape_and_transform_to_scores(y_pred)

        score = -(self.a * y_true + (1 - self.a) * (1 - y_true)) * ((1 - (y_true * p + (1 - y_true) * (1 - p)))
                                                                    ** self.g) * (y_true * np.log(p) + (1 - y_true)
                                                                                  * np.log(1 - p))

        return "focal_loss", np.mean(score), False

    def _independent_api(self, y_pred: np.ndarray, data: lgbm.basic.Dataset) -> Tuple[str, float, bool]:
        y_true = data.get_label()
        p = check_shape_and_transform_to_scores(y_pred)

        score = -(self.a * y_true + (1 - self.a) * (1 - y_true)) * ((1 - (y_true * p + (1 - y_true) * (1 - p)))
                                                                    ** self.g) * (y_true * np.log(p) + (1 - y_true)
                                                                                  * np.log(1 - p))

        return "focal_loss", np.mean(score), False
