import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        a = -1/y_true.shape[0]
        y_pred = y_pred+10e-9
        out = a*(np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)))
        return round(out, 4)
    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n_s, n_c = y_true.shape
        a = -1/n_s
        y_pred= y_pred + 10e-9
        out = np.sum(y_true * np.log(y_pred)) # n_samples
        out = a * out
        return round(out, 4)
        
