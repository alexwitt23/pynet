"""Optimizers for deeplearning."""

import numpy as np

import pynet

class sgd:
    """Stochastic gradient descent with option of momentum."""
    def __init__(
        self, 
        model: pynet.core.model.Model, 
        lr: float = 1e-3, 
        momentum: float = 0.0, 
        weight_decay: float = 0,
        nesterov = False
    ) -> None:
        self.model = model
        self.weight_decay = weight_decay
        self.lr = lr
        # Misleading name, really should be friction since it
        # represented how quickly to decay previous gradients.
        assert momentum < 1, "Please set momentum [0, 1)."
        self.momentum_decay = momentum

    def step(self, grad: np.ndarray):
        """
        Take the incoming grad from the loss fn and propogate
        through layers.
        """
        self.grad: float = grad
        self.grad_sum: float = 0
        for layer in reversed(self.model.layers):
            self.grad = layer.backprop(self.grad, self.lr, self.weight_decay)
            self.grad_sum += np.sum(self.grad)
        print(self.grad_sum)
        
