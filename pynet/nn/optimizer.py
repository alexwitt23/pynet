""" Optimizers for deeplearning. Broadly speaking there is no one best 
adaptive learning rate that is better than the rest. It can be problem 
specific or depend on the user's familiarity with the algorithim. """

import abc
from typing import Dict

import numpy as np

import pynet


class Optimizer:

    def __init__(self, model) -> None:
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
class SGD(Optimizer):
    """ Stochastic gradient descent with option of momentum and Nesterov. """

    def __init__(
        self,
        model,
        lr: float = 1e-1,
        momentum: float = 0.999,
        weight_decay: float = 0,
    ) -> None:

        super().__init__(model)
        self.model = model
        self.weight_decay = weight_decay
        self.lr = lr

        # Misleading name, really should be friction since it
        # represented how quickly to decay previous gradients.
        assert momentum < 1, "Please set momentum [0, 1)."
        self.momentum_decay = momentum
        self.momentum = None

        # Tell the model to load the optimizers for the trainable layers.
        self.model.initialize(self)

    def update(self, weights: np.ndarray, dw: np.ndarray) -> None:

        if self.momentum is None:
            self.momentum = np.zeros_like(weights)

        # Calculate velocity
        self.momentum = (
            self.momentum_decay * self.momentum + (1 - self.momentum_decay) * dw
        )
        if weights is not None:
            # Update this layer
            weights -= (self.momentum) * self.lr


class AdaGrad(Optimizer):
    """ Optimizer which individually adapts learning rates for model params 
    by scaling inversely to square root of past grad values.  """

    def __init__(self, model, lr: float = 1e-1) -> None:

        super().__init__(model)
        self.lr = lr
        self.model = model
        # Var which will keep the sum of squared grads for each param
        self.grad_history = None
        self.epsilon = 1e-7  # For numerical stability
        self.model.initialize(self)

    def update(self, weights: np.ndarray, dw: np.ndarray) -> None:

        if self.grad_history is None:
            self.grad_history = np.zeros_like(weights)

        self.grad_history += np.square(dw)  

        if weights is not None:
            weights -= self.lr * dw / (self.epsilon + np.sqrt(self.grad_history))

# TODO(alex) add Nesterov
class RMSProp(Optimizer):
    """ (Hinton, 2012) Adapts AdaGrad's adgrad gradient accumulation to exponentially 
    weighted moving average. This lessens the blow of adapting large gradients too 
    quickly at the beginning of training. """

    def __init__(self, model, lr: float = 1e-1, decay: float = .99) -> None:

        super().__init__(model)
        self.lr = lr 
        self.decay = decay
        self.model = model
        self.epsilon = 1e-7  # For numerical stability
        self.grad_history = None

        self.model.initialize(self)

    def update(self, weights: np.ndarray, dw: np.ndarray) -> None:

        if self.grad_history is None:
            self.grad_history = np.zeros_like(weights)

        self.grad_history = self.grad_history * self.decay + (1 - self.decay) * np.square(dw)  

        if weights is not None:
            weights -= self.lr * dw / (self.epsilon + np.sqrt(self.grad_history))


class Adam(Optimizer):
    """ (Kingma and Ba, 2014) Adam can be described as a variant on RMSProp and momentum. """
    def __init__(self, model, lr: float = 1e-3, rho1: float = .9, rho2: float = .999) -> None:

        super().__init__(model)
        self.lr = 0.001
        self.delta = 1e-8
        self.rho1 = rho1
        self.rho2 = rho2

        # Initialize the 1st and 2nd moments to None
        self.s = None
        self.r = None

        self.model.initialize(self)

    def update(self, weights: np.ndarray, dw: np.ndarray) -> None:

        if self.s is None:
            self.s = np.zeros_like(weights)
            self.r = np.zeros_like(weights)

        self.s = self.rho1 * self.s + (1 - self.rho1) * dw 
        self.r = self.rho2 * self.r + (1 - self.rho2) * np.square(dw)

        # The following operations help to jump the moments since both
        # start out as 0.
        # Correct bias in first moment estimate
        s_hat = self.s / (1 - self.rho1)
        # Correct bias in second moment estimate
        r_hat = self.r / (1 - self.rho2)

        if weights is not None:
            weights -= self.lr * s_hat / (self.delta + np.sqrt(r_hat))