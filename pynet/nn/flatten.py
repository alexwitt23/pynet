"""Simple layer that flattens input."""

import numpy as np


class Flatten:
    def __init__(self) -> None:
        self.weights = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Assume batch is first dimension and grayscale.
        x.shape = [batch, W, H]."""
        dims = x.shape
        return x.reshape(dims[0], dims[1] * dims[2])

    # TODO clean this up.
    def backprop(self, dx: np.ndarray) -> None:
        return dx

    def update(self, grad: np.ndarray, lr: float, decay: float) -> None:
        pass
