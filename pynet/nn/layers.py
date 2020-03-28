"""Contains all the layers in this library."""

from typing import Tuple

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
        return np.dot(dout, self.weights.transpose())


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
        kernel_size: Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = in_channels
        self.num_filters = out_channels
        self.kernel_size = kernel_size  # (W, H)
        self.stride = stride
        self.padding = padding

        # Create numpy filter volume
        self.kernel = np.random.randn(kernel_size[0], kernel_size[1], out_channels)

        if bias:
            self.biases = np.zeros((1, out_channels))
        else:
            self.biases = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through convolutional layer. This will be implemented in a 
        slightly confusing fashion, but the idea is to level numpy's matrix math to 
        speed up the calculations."""

        assert len(x.shape) == 4, "Input must be [N, W, H, C]"
        # Get the width and height of the incoming data.
        batch_size, width, height, filters_in = x.shape
        assert filters_in == self.input_size, f"Improper input channles: {x.shape}!"
        # Calculate output filter W, H
        width_out = int((width + self.padding - self.kernel_size[0]) / self.stride + 1)
        height_out = int(
            (height + self.padding - self.kernel_size[1]) / self.stride + 1
        )

        # Equivalent to [x for _ in range(kernel_height) for x in range(kernel_width)]
        i = np.repeat(
            np.tile(np.arange(self.kernel_size[0]), self.kernel_size[1]),
            self.input_size,
        )
        # Equivalent to [y for y in range(kernel_height) for _ in range(kernel_width)]
        j = np.tile(
            np.arange(self.kernel_size[1]), self.kernel_size[0] * self.input_size
        )
        
        # These are the indices of the output width [x for x in range(width_out)]
        # np.repeat turns this in to [x for _ in range(out_height) for x in range(width_out)]
        i1 = self.stride * np.repeat(np.arange(width_out), height_out)

        # This is equivalent to [y for y in range(out_height) for _ in range(width_out)]
        j1 = self.stride * np.tile(np.arange(height_out), width_out)

        i = i.reshape(-1, 1) + i1.reshape(1, -1)
        j = j.reshape(-1, 1) + j1.reshape(1, -1)
        print(i.shape, j.shape)
        # Get indices for each of the incoming filter repeated by the number of weights one of this
        # layer's filters.
        filter_ids = np.repeat(
            np.arange(self.input_size), self.kernel_size[0] * self.kernel_size[1]
        ).reshape(-1, 1)

        self.img_slices = (
            x[:, i, j, filter_ids]
            .transpose(1, 2, 0)
            .reshape(self.kernel_size[0] * self.kernel_size[1] * self.input_size, -1)
        )
        # Flatten out each filter into a column
        self.weights_col = np.repeat(
            self.kernel.reshape((self.num_filters, -1)), self.input_size, axis=1
        )
        retval = np.dot(self.weights_col, self.img_slices)  # (num_filters_out, out_w * out_h * batch_size)
        retval = retval.reshape(((self.num_filters, width_out, height_out, batch_size)))
        
        return retval.transpose(3, 1, 2, 0)

    def backwards(self, dout: np.ndarray) -> np.ndarray:
        
        # Note the similarity to the dense layer:
        return np.dot(dout, self.weights_col.transpose())
        

    def input_size(self):
        pass

    def parameters(self) -> int:
        return self.output_size * self.kernel_size[0] * self.kernel_size[1]

    def output_size(self) -> int:
        return self.num_filters

    def update(self):
        return None


class LogSoftmax():
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


class Flatten:
    def __init__(self) -> None:
        self.weights = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Assume batch is first dimension and grayscale.
        x.shape = [batch, W, H]."""
        dims = x.shape
        return x.flatten()

    # TODO clean this up.
    def backprop(self, dx: np.ndarray) -> None:
        return dx

    def update(self, grad: np.ndarray, lr: float, decay: float) -> None:
        pass
