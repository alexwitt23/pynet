"""Softmax layer."""

import numpy as np 


class softmax:
    def __init__(self, input_size, axis) -> None:
        self.input_size = input_size
        self.axis = axis
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x), axis=self.axis)

    def backprop(self, x: np.ndarray) -> np.ndarray:
        pass