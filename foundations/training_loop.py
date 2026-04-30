import numpy as np
from numpy.typing import NDArray
from typing import Tuple

class Solution:
    def train(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        epochs: int,
        lr: float
    ) -> Tuple[NDArray[np.float64], float]:

        n_samp, n_features = X.shape

        W = np.zeros(n_features)   # shape: (n_features,)
        b = 0.0

        for it in range(epochs):
            # y_hat: (n_samples,)
            y_hat = X @ W + b

            # error: (n_samples,)
            error = y_hat - y

            # dW: (n_features,)
            dW = (2 / n_samp) * (X.T @ error)

            # db: scalar
            db = (2 / n_samp) * np.sum(error)

            W = W - lr * dW
            b = b - lr * db

        return (np.round(W, 5), round(b, 5))