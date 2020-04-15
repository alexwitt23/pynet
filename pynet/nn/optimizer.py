""" Optimizers for deeplearning. """

import abc
from typing import Dict

import numpy as np

import pynet


class Optimizer:

    def __init__(self, model: pynet.nn.model) -> None:
        self.model = model

    def step(self, dout: np.ndarray) -> None:
        """ Take the incoming grad from the loss fn and propogate through layers. """
        self.dout = dout

        for layer in reversed(self.model.layers):
            
            # Propagate the gradient back through this layer, and get gradietn w.r.t weights
            self.dout = layer.backwards(self.dout)

    @abc.abstractmethod
    def update(self, weights: np.ndarray, dw: np.ndarray) -> None:
        raise NotImplementedError("Must implement update method!")

# TODO (add Nesterov)
class SGD:
    """ Stochastic gradient descent with option of momentum and Nesterov. """

    def __init__(
        self,
        model: pynet.nn.model.Model,
        lr: float = 1e-1,
        momentum: float = 0.0,
        weight_decay: float = 0,
    ) -> None:

        super().__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.lr = lr

        # Misleading name, really should be friction since it
        # represented how quickly to decay previous gradients.
        assert momentum < 1, "Please set momentum [0, 1)."
        self.momentum_decay = momentum
        self.momentum = 0
        # Set the velocity
        self.velocity_dict = {idx: 0 for idx in range(len(self.model.layers) + 1)}

        # Tell the model to load the optimizers for the trainable layers.
        self.model.initialize(self)

    def update(self, weights: np.ndarray, dw: np.ndarray) -> None:

        # Calculate velocity
        self.momentum = (
            self.momentum_decay * self.momentum - (1 - self.momentum_decay) * dw
        )
        if weights is not None:
            # Update this layer
            weights += (self.momentum + weights * self.weight_decay) * self.lr


class AdGrad(Optimizer):
    """ Optimizer which individually adapts learning rates for model params 
    by scaling inversely to sqyare root past grad values.  """

    def __init__(self, model: pynet.nn.model.Model, start_lr: float = 1e-1) -> None:

        super().__init__()
        self.start_lr = start_lr
        self.model = model
        # Var which will keep the sum of squared grads for each param
        self.grad_history = None
        self.model.initialize(self)
        self.epsilon = 1e-7  # For numerical stability

    def update(self, weights: np.ndarray, dw: np.ndarray) -> None:

        if self.grad_history is None:
            self.grad_history = np.zeros_like(weights)

        self.grad_history += np.square(dw)  

        if weights is not None:
            weights += weights * 1 / (self.epsilon + np.sqrt(self.grad_history))
