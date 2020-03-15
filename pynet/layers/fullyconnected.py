"""
Fully connected layer defined by y = Wx + b.
"""

import numpy as np 

from pynet.activations.relu import ReLU
from pynet.core.regularizer import regularize_dict


class Linear:
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

        N: batch size.
        M: Output size of hidden layer.
        D: Input size.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        
        self.input = np.empty([input_size])

        if activation is None:
            self.use_activation = False
        else:
            self.use_activation = True
            self.activation = activation

        self.use_bias = bias
        self.regularization = regularize_dict[regularization]

        if self.use_bias:
            self.biases = np.zeros((1, output_size))
        else:
            self.biases = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.
        
        Args:
            - x: Input Tensor.

        Returns:
            Apply an affine transformation y = Wx + b.
        """
        self.input = x
        self.out = np.dot(self.input, self.weights)
        # Apply bias if included
        if self.use_bias:
            self.out += self.biases
        if self.use_activation:
            self.out = self.activation.apply(self.out)
        return self.out

    def backprop(self, dout: np.ndarray, lr: float, decay: float) -> np.ndarray:
        """
        Backprop of gradient to weights, biases, and chain rule.
        See derivation: http://cs231n.stanford.edu/handouts/linear-backprop.pdf.

        Args:
            dout: The derivative of loss w.r.t this layer's output. (N, M)
            lr: learning rate 
            decay: the weight decay value to apply.
        
        Returns:
            Derivative of loss w.r.t this layer's input.
        """

        if self.use_activation:
            dout = self.activation.backprop(dout, self.out)
            
        # Backprop to input for this layer
        dinput = np.dot(dout, self.weights.transpose())
        
        # Adjust weights
        self.weights -= (
            np.dot(self.input.transpose(), dout) * lr 
        ) 
        
        if decay > 0:
            self.weights -= self.regularization.apply(self.weights, decay) * lr
        
        if self.use_bias:
            self.biases -= (dout.sum(axis=0) * lr)
        
        return dinput

