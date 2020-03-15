"""Optimizers for deeplearning."""

import numpy as np

import pynet

class sgd:
    """Stochastic gradient descent with option of momentum."""
    def __init__(
        self, 
        model: pynet.core.model.Model, 
        lr: float = 1e-1, 
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
        self.velocity: float = 0
        self.grad_dict = {}

    def step(self, grad: np.ndarray):
        """
        Take the incoming grad from the loss fn and propogate
        through layers.
        """
        self.grad: float = grad
        self.grad_dict[0] = self.grad
        self.grad_sum: float = 0

        for idx, layer in enumerate(reversed(self.model.layers)):
            self.grad = layer.backprop(self.grad, self.lr, self.weight_decay)
            # Store grad for this layer (reversed order)
            self.grad_dict[idx + 1] = self.grad
            # Accumulate the gradient
            self.grad_sum += np.sum(self.grad)

        # Calculate velocity
        self.velocity = self.momentum_decay * self.velocity - (self.lr * self.grad_sum) 

        # Pass the velocity and grad to layers 
        for idx, layer in enumerate(reversed(self.model.layers)):
            layer.update(self.grad_dict[idx] + self.velocity, self.lr, self.weight_decay)
        
        return None

    def zero_grad(self) -> None:
        """Zeros out the accumulated velocity."""
        self.velocity = 0

        return None
        
