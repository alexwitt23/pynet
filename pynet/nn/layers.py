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
    def update(self, lr: float = 1, gamma: float = 0) -> None:
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


class Linear(Layer):
    """Fully connected layer defined by y = Wx + b."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        """Linear, or fully connected, layer.
        
        Args:
            input_size: number of input units. [N x M]
            output_size: number of output units. [N X D]
            bias: To include the bias vector or not.

        Returns:
            Matrix of size [N x D]
        """
        super().__init__()
        self.input_size = np.array(input_size)
        self.output_size = np.array(output_size)
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

        return self.out

    def backwards(self, dout: np.ndarray) -> np.ndarray:
        """
        Backprop of gradient to weights, biases, and chain rule.
        See derivation: http://cs231n.stanford.edu/handouts/linear-backprop.pdf.

        Args:
            dout: The derivative of loss w.r.t this layer's output. (N, M)
        
        Returns:
            Derivative of loss w.r.t this layer's input.
        """
        self.grad = dout
        # Backprop to input for this layer
        dinput = np.dot(dout, self.weights.transpose())

        return dinput

    def update(self, lr: float = 1, decay: float = 0) -> None:
        """Takes in the amount to update weights by. Input given by optimzers."""
        # Adjust weights
        self.weights += np.dot(self.input.transpose(), self.grad)
        """
        if decay > 0:
            self.weights -= self.regularization.apply(self.weights, decay) * lr
        """
        if self.use_bias:
            self.biases += self.grad.sum(axis=0)

        return None

    def parameters(self) -> int:
        return self.input_size * self.output_size
    
    def input_size(self) -> np.ndarray:
        return self.input_size

    def output_size(self) -> np.ndarray:
        return self.output_size


class Conv2D(Layer):
    """Two dimensional convolutional layer."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = in_channels
        self.output_size = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

        # Create numpy filter volume
        self.kernel = np.random.randn(kernel_size, kernel_size, out_channels)

        if self.use_bias:
            self.biases = np.zeros((1, out_channels))
        else:
            self.biases = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # Pad the input
        self.input = np.pad(x, (self.padding,))
        print(self.input.shape)
        # Don't know HW of input until here..
        self.output_shape = int(
            (1 / self.stride) * (2 * self.padding + x.shape[1] - self.kernel_size) + 1
        )
        # NWHC
        self.output = np.zeros(
            (x.shape[0], self.output_shape, self.output_shape, self.output_size)
        )
        assert len(x.shape) == 4
        # Loop over the batch
        for item in range(x.shape[0]):
            # Loop over the filters
            for f in range(self.output_size):
                # Apply fiter to entire volume and loop over the input
                for j in range(0, x.shape[1] - self.kernel_size, self.stride):
                    for k in range(0, x.shape[2] - self.kernel_size, self.stride):
                        # Multiply input by filter
                        self.output[item, j, k, f] = np.sum(
                            np.multiply(
                                x[item, j + self.kernel_size, k + self.kernel_size, :],
                                self.kernel[:, :, f],
                            )
                        )

                if self.use_bias:
                    self.output += self.biases[0, f]

        return self.output



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
