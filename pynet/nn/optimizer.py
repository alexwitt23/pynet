"""Optimizers for deeplearning."""

from typing import Dict

import numpy as np

import pynet


class sgd:
    """Stochastic gradient descent with option of momentum and Nesterov."""

    def __init__(
        self,
        model,
        lr: float = 1e-1,
        momentum: float = 0.0,
        weight_decay: float = 0,
        nesterov=False,
    ) -> None:
        self.model = model
        self.weight_decay = weight_decay
        self.lr = lr
        self.nesterov: bool = nesterov

        # Misleading name, really should be friction since it
        # represented how quickly to decay previous gradients.
        assert momentum < 1, "Please set momentum [0, 1)."
        self.momentum_decay = momentum

        # Set the velocity
        self.zero_grad()

    def step(self, grad: np.ndarray) -> None:
        """Take the incoming grad from the loss fn and propogate through layers."""
        self.grad: np.ndarray = grad * self.lr
        self.velocity_dict[0]: np.ndarray = grad
        for idx, layer in enumerate(reversed(self.model.layers)):
            
            # Apply nesterov momentum (_correction factor_)
            if self.nesterov:
                # Make sure the layer has weights and we aren't on first step of accumulation
                if layer.weights is not None and self.velocity_dict[idx] is not 0:
                    layer.weights += self.momentum_decay * np.average(
                        self.velocity_dict[idx], axis=0
                    )
                    
            # Apply L2 weight regularization
            if layer.weights is not None:
                layer.weights -= layer.weights * self.weight_decay
            
            self.grad = layer.backwards(self.grad)
            # Calculate velocity
            
            self.velocity_dict[idx + 1] = self.momentum_decay * self.velocity_dict[
                idx + 1
            ] - (self.lr * self.grad)
            
            layer.update(self.velocity_dict[idx])

    def zero_grad(self) -> None:
        """Zeros out the accumulated velocity."""
        self.velocity_dict = {idx: 0 for idx in range(len(self.model.layers) + 1)}
