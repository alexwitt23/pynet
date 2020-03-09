"""
Fully connected layer defined by y = Wx + b.
"""

import numpy as np 

from pynet.activations.relu import ReLU
from pynet.core.regularizer import regularize_dict


class linear:
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        activation = ReLU, 
        regularization: str = "l2",
        bias: bool = False
    ) -> None:
        """
        Given input size m, and output size n, create [m X n] matrix 
        of weights.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size, output_size) * 0.01
        self.input = np.empty([input_size])

        if activation is None:
            self.use_activation = False
        else:
            self.use_activation = True
            self.activation = activation

        self.use_bias = bias
        self.regularization = regularize_dict[regularization]

        if self.use_bias:
            self.biases = np.random.rand(1, output_size) * 0.01
        else:
            self.biases = np.zeros((1, output_size))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.input = x
        self.out = np.dot(self.input, self.weights)
        if self.use_bias:
            self.out += self.biases
        return self.out

    def backprop(self, x: np.ndarray, lr: float, decay: float) -> np.ndarray:
        """
        Backprop of gradient to weights, biases, and chain rule.
        See derivation: http://cs231n.stanford.edu/handouts/linear-backprop.pdf
        """
        if self.use_activation:
            x = self.activation.backprop(x)
        # Backprop to input for this layer
        dinput = np.dot(x, self.weights.transpose())

        # Adjust weights
        self.weights -= (
            np.dot(self.input.transpose(), x) * lr 
        ) 

        if decay > 0:
            self.weights -= self.regularization.apply(self.weights, decay) * lr

        if self.use_bias:
            self.biases -= (x.sum(axis=0) * lr)
        
        return dinput

