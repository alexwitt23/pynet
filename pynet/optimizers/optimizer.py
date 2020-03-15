"""Attempt at an optimizer."""

import numpy as np


class sgd:
    def __init__(self, model, lr: float = 1e-3, weight_decay: float = 0) -> None:
        self.model = model
        self.weight_decay = weight_decay
        self.lr = lr

    def step(self, grad: np.ndarray):
        self.grad = grad
        for layer in reversed(self.model.layers):
            self.grad = layer.backprop(self.grad, self.lr, self.weight_decay)
