"""Mean Squared Error"""

import numpy as np


class MSE:
    def __init__(self, input_size) -> None:
        self.input_size = input_size

    def __call__(self, pred, target) -> np.float:
        self.target = target
        self.pred = pred
        self.loss = 0.5 * np.mean(np.square(target - pred)) / self.target.shape[1]
        return self.loss

    def backwards(self) -> np.ndarray:
        # Derivatives
        grad = (self.target - self.pred) / self.target.shape[1]
        return grad
