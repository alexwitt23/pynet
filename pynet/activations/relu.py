"""ReLU Activation."""

import numpy as np 


class ReLU:

    @staticmethod
    def apply(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def backprop(x: np.ndarray) -> np.ndarray:
        x[x <= 0] = 0
        return x