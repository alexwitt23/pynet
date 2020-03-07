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
        Given input size m, and output size n, create [n x m] matrix 
        of weights.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size, output_size)
        self.activation = activation()  # Why does this have to be instance?

        if bias:
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

    def backprop(self, x, lr: float = 1e-4) -> np.ndarray:
        """
        Backprop of gradient to weights, biases, and original input.
        See derivation: http://cs231n.stanford.edu/handouts/linear-backprop.pdf
        """
        # Backprop to input for this layer
        dinput = np.matmul(x, self.weights.transpose())
        dinput = np.multiply(dinput, self.activation.backprop(self.input))
        # Adjust weights
        self.weights += (
            np.matmul(self.input.transpose(), x)
        ) * lr 
        self.biases += (x * lr)
        
        return dinput

