"""Mean Squared Error"""

import numpy as np 


class mse:

    def __init__(self, input_size) -> None: 
        self.input_size = input_size

    def __call__(self, pred, target) -> np.float:
        self.target = target
        self.pred = pred
        self.loss =  0.5 * np.mean(np.square(target - pred))
        return self.loss

    def backprop(self) -> np.ndarray:
        # Derivatives
        grad = (self.target - self.pred) / self.target.shape[0]
        return grad


class mse_l2:
    """MSE with L2 weight regularization."""
    def __init__(self, input_size, decay=1e-4) -> None: 
        self.input_size = input_size

    def __call__(self, pred, w, target) -> np.float:
        self.target = target
        self.pred = pred
        self.loss = 0.5 * np.mean(np.square(target - pred) + np.linalg.norm(w))
        return self.loss

    def backprop(self) -> np.ndarray:
        # Derivatives
        grad = (self.pred - self.target) / self.target.shape[0]
        return grad