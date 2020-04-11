"""Optimizers for deeplearning."""

from typing import Dict
import abc

import numpy as np

import pynet


class Optimizer:
    @abc.abstractmethod
    def step(self, dout: np.ndarray) -> None:
        raise NotImplementedError("Must implement step method!")

    @abc.abstractmethod
    def update(self, weights: np.ndarray, dw: np.ndarray) -> None:
        raise NotImplementedError("Must implement update method!")


class SGD:
    """Stochastic gradient descent with option of momentum and Nesterov."""

    def __init__(
        self,
        model: pynet.nn.model.Model,
        lr: float = 1e-1,
        momentum: float = 0.0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ) -> None:
        # Assign the member variables
        self.model = model

        self.weight_decay = weight_decay
        self.lr = lr
        self.nesterov = nesterov

        # Misleading name, really should be friction since it
        # represented how quickly to decay previous gradients.
        assert momentum < 1, "Please set momentum [0, 1)."
        self.momentum_decay = momentum
        self.momentum = 0
        # Set the velocity
        self.velocity_dict = {idx: 0 for idx in range(len(self.model.layers) + 1)}

        # Tell the model to load the optimizers for the trainable layers.
        self.model.initialize(self)

    def step(self, dout: np.ndarray) -> None:
        """Take the incoming grad from the loss fn and propogate through layers."""
        self.dout = dout

        for idx, layer in enumerate(reversed(self.model.layers)):
            """
            # Apply nesterov momentum (_correction factor_)
            if self.nesterov:
                # Make sure the layer has weights and we aren't on first step of accumulation
                if layer.weights is not None and self.velocity_dict[idx] is not 0:
                    layer.update(self.momentum_decay * self.velocity_dict[idx])
            """
            # Propagate the gradient back through this layer, and get gradietn w.r.t weights
            self.dout = layer.backwards(self.dout)

    def update(self, weights: np.ndarray, dw: np.ndarray) -> None:
        # Calculate velocity
        self.momentum = (
            self.momentum_decay * self.momentum - (1 - self.momentum_decay) * dw
        )
        if weights is not None:
            # Update this layer
            weights += (self.momentum + weights * self.weight_decay) * self.lr
