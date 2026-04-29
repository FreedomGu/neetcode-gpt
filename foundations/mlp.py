import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        #print(weights[0].shape)
        x = x.reshape(1,-1)
        for i in range(len(weights)):
            W = np.array(weights[i])
            b = np.array(biases[i])
            print(W.shape, x.shape)
            x = x @ W +b
            x = np.maximum(0, x)
        return np.round(x, 5)[0].tolist()
        #pass
