"""
Fully connected layer defined by y = Wx + b.
"""

import numpy as np 

from  activations.relu import ReLU


class linear:
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        activation = ReLU, 
        bias: bool = False) -> None:
        """
        Given input size m, and output size n, create [m X n] matrix 
        of weights.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size, output_size) / 10
        self.activation = activation()
        self.use_bias = bias
        if self.use_bias:
            self.biases = np.random.rand(1, output_size)
        else:
            self.biases = np.zeros((1, output_size))
        self.input = np.empty([input_size])

    def __call__(self, x) -> np.ndarray:
        """Forward pass."""
        self.input = x
        return  (
                self.activation(
                    np.matmul(self.input, self.weights) + self.biases
                )
        )

    def backprop(self, x: np.ndarray, lr: float, decay: float) -> np.ndarray:
        """
        Backprop of gradient to weights, biases, and original input.
        See derivation: http://cs231n.stanford.edu/handouts/linear-backprop.pdf
        """
        # Backprop to input for this layer
        dinput = np.matmul(x, self.weights.transpose())
        dinput = np.multiply(dinput, self.activation.backprop(self.input))

        # Adjust weights
        self.weights += (np.matmul(self.input.transpose(), x) / self.input.shape[1]) * lr 

        if decay > 0:
            self.weights -= (decay / self.input.shape[1]) * self.weights

        if self.use_bias:
            self.biases += (x * lr) * (1 / self.input.shape[1])
        
        return dinput

