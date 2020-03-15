"""Log Softmax layer.

Same as normal softmax except a log is applied to simplify 
the gradient calculation. 

Outputs value in range [-1, 0)

d/dx log(softmax(x)) = 1 - softmax(x)
"""

import numpy as np


class LogSoftmax:
    def __init__(self, input_size, axis) -> None:
        self.input_size = input_size
        self.axis = axis
        self.num_params = 0
        self.weights = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        inter = np.exp(x)  # - np.amax(x, axis=self.axis, keepdims=True)

        self.out = np.log(inter) - np.log(inter.sum(axis=self.axis, keepdims=True))
        self.softmax = np.exp(self.out)

        return self.out

    def backprop(self, x: np.ndarray, lr: float, weight_decay: float) -> np.ndarray:
        # Apply gradient, which is 1 - p(x) where x = target.
        # Then complete chain rule with incoming gradient
        self.softmax += np.multiply(1, x)
        return self.softmax / self.input.shape[0]
