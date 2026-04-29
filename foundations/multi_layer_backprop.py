import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        x1 =  x @ W1.T + b1 # x1 # batch,2 # W1 # 2,2 # W2 1,2 
        x1_a = np.maximum(0, x1)
        y_hat = x1_a @ W2.T + b2 # 1,2 2,1 

        loss = np.mean((y_hat - y_true) ** 2)
        n = len(y_true) if y_true.ndim > 0 else 1
        dz2 = 2* (y_hat - y_true)/n
        print("!:", y_hat.shape, x1_a.shape)
        dw2 = dz2.reshape(-1, 1) @ x1_a.reshape(1, -1)
        db2 = dz2

        # dw2 should be match with 
        da1 = dz2.reshape(1, -1)@ W2
        da1 = da1.flatten()
        dz1 = da1 = da1 * (x1_a >0).astype(float)
        dW1 = dz1.reshape(-1, 1) @ x.reshape(1, -1)
        db1 = dz1
        
        return {'loss':np.round(loss, 4).tolist(),
                'dW2':np.round(dw2, 4).tolist(),
                'db2':np.round(db2, 4).tolist(),
                'dW1':np.round(dW1, 4).tolist(),
                'db1':np.round(db1, 4).tolist()
                }


