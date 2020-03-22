"""Contains all the layers in this library."""

import abc
import numpy as np


class Layer(abc.ABC):
    """Define a collection of member functions that must be 
    implemented by all layers."""

    @abc.abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Must override __call__ method!")

    @abc.abstractmethod
    def backwards(self, dout: np.ndarray) -> np.ndarray:
        """Send the gradient through this layer and update weights."""
        raise NotImplementedError("Must override backprop method!")

    @abc.abstractmethod
    def parameters(self) -> float:
        """Number of trainable params in the model."""
        raise NotImplementedError("Must implement the number of layer parameters!")

    @abc.abstractmethod
    def input_size(self) -> np.ndarray:
        """Size of layer's input size."""
        raise NotImplementedError("Implement input_size!")

    @abc.abstractmethod
    def output_size(self) -> np.ndarray:
        """Size of layer's output size."""
        raise NotImplementedError("Implement output_size!")


"""Fully connected layer defined by y = Wx + b."""


class Linear(Layer):
    def __init__(self, input_size: int, output_size: int, bias: bool = False,) -> None:
        super().__init__()
        """Linear, or fully connected, layer.
        
        Args:
            input_size: number of input units. [N x M]
            output_size: number of output units. [N X D]
            bias: To include the bias vector or not.

        Returns:
            Matrix of size [N x D]
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.input = np.empty([input_size])
        self.num_params = input_size * output_size

        self.use_bias = bias

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

    def backprop(self, dout: np.ndarray) -> np.ndarray:
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

        return dinput

    def update(self, grad: np.ndarray, lr: float, decay: float) -> None:
        """Takes in the amount to update weights by. Input given by optimzers."""
        # Adjust weights
        self.weights += np.dot(self.input.transpose(), grad)

        if decay > 0:
            self.weights -= self.regularization.apply(self.weights, decay) * lr

        if self.use_bias:
            self.biases += grad.sum(axis=0)

        return None


class LogSoftmax(Layer):
    """Apply log to softmax for more preferable numerical properties."""

    def __init__(self, input_size, axis) -> None:
        super().__init__()
        self.input_size = input_size
        self.axis = axis
        self.num_params = 0
        self.weights = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        inter = np.exp(x)  # - np.amax(x, axis=self.axis, keepdims=True)

        self.out = np.log(inter) - np.log(inter.sum(axis=self.axis, keepdims=True))
        self.softmax = np.exp(self.out)

        return self.out

    def backprop(self, x: np.ndarray) -> np.ndarray:
        # Apply gradient, which is 1 - p(x) where x = target.
        # Then complete chain rule with incoming gradient
        self.softmax += np.multiply(1, x)
        return self.softmax / self.input.shape[0]

    def update(self, grad, lr, weight_decay) -> None:
        """No params for logsoftmax."""
        return None
