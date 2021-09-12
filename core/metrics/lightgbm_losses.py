from typing import Tuple
import numpy as np
from scipy.misc import derivative
import lightgbm as lgbm


class BinaryFocalLossLightgbm:
    """
    Focal Loss paper: https://arxiv.org/abs/1708.02002
    Loss based on this implementation: https://github.com/jrzaurin/LightGBM-with-Focal-Loss
    Focal Loss for binary classification task with unbalanced target
    """
    def __init__(self, alpha: float, gamma: float) -> None:
        self.a = alpha
        self.g = gamma

    def __call__(self, arg1, arg2) -> Tuple[np.ndarray, np.ndarray]:
        y_true: np.ndarray = None
        y_pred: np.ndarray = None
        if isinstance(arg1, np.ndarray) and isinstance(arg2, np.ndarray):
            y_true = arg1
            y_pred = arg2
        elif isinstance(arg1, np.ndarray) and isinstance(arg2, lgbm.basic.Dataset):
            y_true = arg2.get_label()
            y_pred = arg1

        partial_fl = lambda x: self.fl(x, y_true)
        grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
        hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
        return grad, hess

    def fl(self, x, t):
        p = 1 / (1 + np.exp(-x))
        return -(self.a * t + (1 - self.a) * (1 - t)) * ((1 - (t * p + (1 - t) * (1 - p))) ** self.g) * (
                    t * np.log(p) + (1 - t) * np.log(1 - p))
