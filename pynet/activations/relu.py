"""ReLU Activation."""

import numpy as np 


class ReLU:

    @staticmethod
    def __call__(x: np.ndarray) -> None:
        return np.maximum(x, 0)

    @staticmethod
    def backprop(x: np.ndarray) -> None:
        return np.heaviside(x, 0)